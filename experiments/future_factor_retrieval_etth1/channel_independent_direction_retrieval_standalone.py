"""
Standalone channel-independent retrieval experiment with same-direction loss.

This is the simplified current main experiment.  It does not import helpers from
older experiment folders.  For each channel, it retrieves from the full train
bank in target batches and scores candidates with:

    shape_weight * row_median_norm(shape_mse)
      + direction_weight * row_median_norm(same_direction_loss)
      + raw_weight * row_median_norm(raw_mse)
      + std_weight * row_median_norm(std_log_ratio_loss)

The reported baseline uses shape_mse only.  The rerank result uses the combined
score above.  Futures are aligned by the selected history window statistics
before inverse-distance averaging.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data_loader import (
    FLAGS,
    collect_windows,
    data_provider,
    build_shape_features,
)


EPS = 1e-6


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("weights must contain at least one value")
    if any(value < 0.0 for value in values):
        raise ValueError("weights must be non-negative")
    return values


def fmt_float(value: float) -> str:
    return f"{value:g}"


def select_device(backend: str) -> torch.device | None:
    if backend == "numpy":
        return None
    if backend in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if backend in ("auto", "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if backend == "cpu":
        return torch.device("cpu")
    if backend in ("cuda", "mps"):
        print(f"requested backend={backend!r} is unavailable; falling back to torch CPU")
        return torch.device("cpu")
    if backend == "auto":
        return torch.device("cpu")
    raise ValueError(f"unsupported backend: {backend}")


def direction_parts(windows: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    delta = np.diff(windows, axis=1).reshape(windows.shape[0], -1)
    pos = (delta > eps).astype(np.float32, copy=False)
    neg = (delta < -eps).astype(np.float32, copy=False)
    nz = pos + neg
    return {
        "pos": pos,
        "neg": neg,
        "nz": nz,
        "count": nz.sum(axis=1).astype(np.float32, copy=False),
    }


def squared_dist_batch(query: np.ndarray, bank: np.ndarray, bank_t: np.ndarray, bank_sq: np.ndarray) -> np.ndarray:
    denom = max(1, bank.shape[1])
    query_sq = np.sum(query * query, axis=1, keepdims=True)
    dist = query_sq + bank_sq[None, :] - 2.0 * (query @ bank_t)
    return np.maximum(dist / denom, 0.0).astype(np.float32, copy=False)


def std_loss_batch(query_std: np.ndarray, train_std: np.ndarray) -> np.ndarray:
    ratio = query_std[:, None, :] / train_std[None, :, :]
    return np.mean(np.log(ratio) ** 2, axis=2).astype(np.float32, copy=False)


def direction_loss_batch(query_parts: dict[str, np.ndarray], train_parts: dict[str, np.ndarray]) -> np.ndarray:
    same = query_parts["pos"] @ train_parts["pos_t"] + query_parts["neg"] @ train_parts["neg_t"]
    both_nonzero = query_parts["nz"] @ train_parts["nz_t"]
    comparable = query_parts["count"][:, None] + train_parts["count"][None, :] - both_nonzero
    agreement = same / np.maximum(comparable, 1.0)
    return np.where(comparable > 0.0, 1.0 - agreement, 0.0).astype(np.float32, copy=False)


def row_median_normalize(dist: np.ndarray) -> np.ndarray:
    scale = np.median(dist, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return (dist / scale).astype(np.float32, copy=False)


def select_topk(score: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_k, score.shape[1])
    partial = np.argpartition(score, k - 1, axis=1)[:, :k]
    partial_score = np.take_along_axis(score, partial, axis=1)
    order = np.argsort(partial_score, axis=1)
    selected_idx = np.take_along_axis(partial, order, axis=1).astype(np.int32, copy=False)
    selected_score = np.take_along_axis(partial_score, order, axis=1).astype(np.float32, copy=False)
    return selected_idx, selected_score


def torch_row_median_normalize(dist: torch.Tensor) -> torch.Tensor:
    scale = torch.median(dist, dim=1, keepdim=True).values
    scale = torch.where(torch.isfinite(scale) & (scale > EPS), scale, torch.ones_like(scale))
    return dist / scale


def torch_squared_dist_batch(
    query: torch.Tensor,
    bank_t: torch.Tensor,
    bank_sq: torch.Tensor,
    denom: int,
) -> torch.Tensor:
    query_sq = torch.sum(query * query, dim=1, keepdim=True)
    dist = query_sq + bank_sq.unsqueeze(0) - 2.0 * (query @ bank_t)
    return torch.clamp(dist / max(1, denom), min=0.0)


def torch_std_loss_batch(query_std: torch.Tensor, train_std: torch.Tensor) -> torch.Tensor:
    ratio = query_std[:, None, :] / train_std[None, :, :]
    return torch.mean(torch.log(ratio) ** 2, dim=2)


def torch_direction_loss_batch(
    query_pos: torch.Tensor,
    query_neg: torch.Tensor,
    query_nz: torch.Tensor,
    query_count: torch.Tensor,
    train_pos_t: torch.Tensor,
    train_neg_t: torch.Tensor,
    train_nz_t: torch.Tensor,
    train_count: torch.Tensor,
) -> torch.Tensor:
    same = query_pos @ train_pos_t + query_neg @ train_neg_t
    both_nonzero = query_nz @ train_nz_t
    comparable = query_count[:, None] + train_count[None, :] - both_nonzero
    agreement = same / torch.clamp(comparable, min=1.0)
    return torch.where(comparable > 0.0, 1.0 - agreement, torch.zeros_like(agreement))


def torch_channel_shape_features(windows: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(windows, dtype=torch.float32, device=device)
    mean = torch.mean(x, dim=1, keepdim=True)
    std = torch.clamp(torch.std(x, dim=1, keepdim=True, unbiased=False), min=EPS)
    return (x - mean) / std


def torch_channel_direction_parts(windows: np.ndarray, eps: float, device: torch.device) -> dict[str, torch.Tensor]:
    x = torch.as_tensor(windows, dtype=torch.float32, device=device)
    delta = x[:, 1:, :] - x[:, :-1, :]
    pos = (delta > eps).to(torch.float32)
    neg = (delta < -eps).to(torch.float32)
    nz = pos + neg
    return {
        "pos": pos,
        "neg": neg,
        "nz": nz,
        "count": torch.sum(nz, dim=1),
    }


def torch_channel_dist_batch(
    query: torch.Tensor,
    bank: torch.Tensor,
    bank_sq: torch.Tensor,
) -> torch.Tensor:
    query_sq = torch.sum(query * query, dim=1)
    dot = torch.einsum("bsc,nsc->bnc", query, bank)
    dist = query_sq[:, None, :] + bank_sq[None, :, :] - 2.0 * dot
    return torch.clamp(dist / max(1, query.shape[1]), min=0.0)


def torch_channel_direction_loss_batch(
    query_parts: dict[str, torch.Tensor],
    train_parts: dict[str, torch.Tensor],
) -> torch.Tensor:
    same = (
        torch.einsum("bsc,nsc->bnc", query_parts["pos"], train_parts["pos"])
        + torch.einsum("bsc,nsc->bnc", query_parts["neg"], train_parts["neg"])
    )
    both_nonzero = torch.einsum("bsc,nsc->bnc", query_parts["nz"], train_parts["nz"])
    comparable = query_parts["count"][:, None, :] + train_parts["count"][None, :, :] - both_nonzero
    agreement = same / torch.clamp(comparable, min=1.0)
    return torch.where(comparable > 0.0, 1.0 - agreement, torch.zeros_like(agreement))


def torch_channel_row_median_normalize(dist: torch.Tensor) -> torch.Tensor:
    scale = torch.median(dist, dim=1, keepdim=True).values
    scale = torch.where(torch.isfinite(scale) & (scale > EPS), scale, torch.ones_like(scale))
    return dist / scale


def torch_channel_std_loss_batch(query_std: torch.Tensor, train_std: torch.Tensor) -> torch.Tensor:
    ratio = query_std[:, None, :] / train_std[None, :, :]
    return torch.log(ratio) ** 2


def inverse_distance_weights(score: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.maximum(score, EPS)
    return (inv / np.maximum(inv.sum(axis=1, keepdims=True), EPS)).astype(np.float32, copy=False)


def softmax_negative_weights(score: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), EPS)
    logits = -score / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits).astype(np.float32, copy=False)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def weights_from_score(score: np.ndarray, weight_mode: str, temperature: float) -> np.ndarray:
    if weight_mode == "inverse":
        return inverse_distance_weights(score)
    if weight_mode == "softmax":
        return softmax_negative_weights(score, temperature)
    if weight_mode == "uniform":
        return np.full_like(score, 1.0 / score.shape[1], dtype=np.float32)
    raise ValueError(f"unsupported weight mode: {weight_mode}")


def align_selected_futures(
    futures: np.ndarray,
    train_windows: np.ndarray,
    target_windows: np.ndarray,
    selected_idx: np.ndarray,
    align_mode: str,
    std_eps: float,
    std_ratio_min: float,
    std_ratio_max: float,
) -> np.ndarray:
    if align_mode == "none":
        return futures

    train_mean = train_windows.mean(axis=1).astype(np.float32, copy=False)
    train_std = np.maximum(train_windows.std(axis=1), std_eps).astype(np.float32, copy=False)
    query_mean = target_windows.mean(axis=1).astype(np.float32, copy=False)
    query_std = np.maximum(target_windows.std(axis=1), std_eps).astype(np.float32, copy=False)

    neighbor_mean = train_mean[selected_idx]
    neighbor_std = train_std[selected_idx]
    if align_mode == "mean":
        return futures - neighbor_mean[:, :, None, :] + query_mean[:, None, None, :]
    if align_mode == "clipped_std":
        ratio = np.clip(query_std[:, None, :] / neighbor_std, std_ratio_min, std_ratio_max)
        return (futures - neighbor_mean[:, :, None, :]) * ratio[:, :, None, :] + query_mean[:, None, None, :]
    if align_mode == "std":
        return (
            (futures - neighbor_mean[:, :, None, :]) / neighbor_std[:, :, None, :]
        ) * query_std[:, None, None, :] + query_mean[:, None, None, :]
    raise ValueError(f"unsupported align mode: {align_mode}")


def predict_selected(
    train_windows: np.ndarray,
    train_preds: np.ndarray,
    target_windows: np.ndarray,
    selected_idx: np.ndarray,
    selected_score: np.ndarray,
    args,
) -> np.ndarray:
    weights = weights_from_score(selected_score, args.weight_mode, args.temperature)
    futures = train_preds[selected_idx]
    futures = align_selected_futures(
        futures,
        train_windows,
        target_windows,
        selected_idx,
        args.align_mode,
        args.std_eps,
        args.std_ratio_min,
        args.std_ratio_max,
    )
    return np.einsum("bk,bkpc->bpc", weights, futures, optimize=True).astype(np.float32)


def metric(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float, float, float]:
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true))
    mspe = np.mean(np.square((pred - true) / true))
    return float(mae), float(mse), float(rmse), float(mape), float(mspe)


def metric_row(setting: str, flag: str, method: str, metrics: tuple[float, ...], extra: dict) -> dict:
    mae, mse, rmse, mape, mspe = metrics
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "setting": setting,
        "flag": flag,
        "method": method,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
    }
    row.update(extra)
    return row


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_channel(
    train_windows_all: np.ndarray,
    train_preds_all: np.ndarray,
    target_windows_all: np.ndarray,
    target_preds_all: np.ndarray,
    channel_idx: int,
    args,
    direction_weights: list[float],
    channel_name: str,
    device: torch.device | None,
) -> dict:
    train_windows = train_windows_all[:, :, channel_idx:channel_idx + 1]
    train_preds = train_preds_all[:, :, channel_idx:channel_idx + 1]
    target_windows = target_windows_all[:, :, channel_idx:channel_idx + 1]
    target_preds = target_preds_all[:, :, channel_idx:channel_idx + 1]

    train_shape = build_shape_features(train_windows)
    target_shape = build_shape_features(target_windows)
    train_shape_t = np.ascontiguousarray(train_shape.T)
    train_shape_sq = np.sum(train_shape * train_shape, axis=1).astype(np.float32, copy=False)

    train_raw = train_windows.reshape(train_windows.shape[0], -1).astype(np.float32, copy=False)
    target_raw = target_windows.reshape(target_windows.shape[0], -1).astype(np.float32, copy=False)
    train_raw_t = np.ascontiguousarray(train_raw.T)
    train_raw_sq = np.sum(train_raw * train_raw, axis=1).astype(np.float32, copy=False)

    train_std = np.maximum(train_windows.std(axis=1), args.std_eps).astype(np.float32, copy=False)
    target_std = np.maximum(target_windows.std(axis=1), args.std_eps).astype(np.float32, copy=False)

    train_dir = direction_parts(train_windows, args.std_eps)
    train_dir["pos_t"] = np.ascontiguousarray(train_dir["pos"].T)
    train_dir["neg_t"] = np.ascontiguousarray(train_dir["neg"].T)
    train_dir["nz_t"] = np.ascontiguousarray(train_dir["nz"].T)
    target_dir = direction_parts(target_windows, args.std_eps)

    if device is not None:
        train_shape_torch = torch.as_tensor(train_shape, dtype=torch.float32, device=device)
        target_shape_torch = torch.as_tensor(target_shape, dtype=torch.float32, device=device)
        train_shape_t_torch = train_shape_torch.T.contiguous()
        train_shape_sq_torch = torch.sum(train_shape_torch * train_shape_torch, dim=1)

        train_raw_torch = torch.as_tensor(train_raw, dtype=torch.float32, device=device)
        target_raw_torch = torch.as_tensor(target_raw, dtype=torch.float32, device=device)
        train_raw_t_torch = train_raw_torch.T.contiguous()
        train_raw_sq_torch = torch.sum(train_raw_torch * train_raw_torch, dim=1)

        train_std_torch = torch.as_tensor(train_std, dtype=torch.float32, device=device)
        target_std_torch = torch.as_tensor(target_std, dtype=torch.float32, device=device)

        train_pos_t = torch.as_tensor(train_dir["pos_t"], dtype=torch.float32, device=device)
        train_neg_t = torch.as_tensor(train_dir["neg_t"], dtype=torch.float32, device=device)
        train_nz_t = torch.as_tensor(train_dir["nz_t"], dtype=torch.float32, device=device)
        train_count = torch.as_tensor(train_dir["count"], dtype=torch.float32, device=device)
        target_pos = torch.as_tensor(target_dir["pos"], dtype=torch.float32, device=device)
        target_neg = torch.as_tensor(target_dir["neg"], dtype=torch.float32, device=device)
        target_nz = torch.as_tensor(target_dir["nz"], dtype=torch.float32, device=device)
        target_count = torch.as_tensor(target_dir["count"], dtype=torch.float32, device=device)

    backend_name = "numpy" if device is None else str(device)
    print(
        f"  {channel_name}: samples={len(target_windows)}, "
        f"candidates={len(train_windows)}, backend={backend_name}",
        flush=True,
    )

    baseline_batches = []
    rerank_batches = {weight: [] for weight in direction_weights}
    for start in range(0, len(target_windows), args.selected_batch_size):
        end = min(start + args.selected_batch_size, len(target_windows))

        if device is None:
            shape_dist = squared_dist_batch(
                target_shape[start:end],
                train_shape,
                train_shape_t,
                train_shape_sq,
            )
            baseline_idx, baseline_score = select_topk(shape_dist, args.top_k)
        else:
            with torch.no_grad():
                shape_dist_t = torch_squared_dist_batch(
                    target_shape_torch[start:end],
                    train_shape_t_torch,
                    train_shape_sq_torch,
                    train_shape.shape[1],
                )
                baseline_score_t, baseline_idx_t = torch.topk(
                    shape_dist_t,
                    min(args.top_k, shape_dist_t.shape[1]),
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                baseline_idx = baseline_idx_t.cpu().numpy().astype(np.int32, copy=False)
                baseline_score = baseline_score_t.cpu().numpy().astype(np.float32, copy=False)
        baseline_batches.append(
            predict_selected(
                train_windows,
                train_preds,
                target_windows[start:end],
                baseline_idx,
                baseline_score,
                args,
            )
        )

        if device is None:
            raw_norm = None
            if args.raw_weight:
                raw_dist = squared_dist_batch(
                    target_raw[start:end],
                    train_raw,
                    train_raw_t,
                    train_raw_sq,
                )
                raw_norm = row_median_normalize(raw_dist)

            std_norm = None
            if args.std_weight:
                std_norm = row_median_normalize(std_loss_batch(target_std[start:end], train_std))

            dir_query = {key: value[start:end] for key, value in target_dir.items()}
            direction_norm = row_median_normalize(direction_loss_batch(dir_query, train_dir))
            shape_norm = row_median_normalize(shape_dist)

        else:
            with torch.no_grad():
                shape_norm = torch_row_median_normalize(shape_dist_t)
                raw_norm = None
                if args.raw_weight:
                    raw_dist_t = torch_squared_dist_batch(
                        target_raw_torch[start:end],
                        train_raw_t_torch,
                        train_raw_sq_torch,
                        train_raw.shape[1],
                    )
                    raw_norm = torch_row_median_normalize(raw_dist_t)

                std_norm = None
                if args.std_weight:
                    std_norm = torch_row_median_normalize(
                        torch_std_loss_batch(target_std_torch[start:end], train_std_torch)
                    )

                direction_norm = torch_row_median_normalize(
                    torch_direction_loss_batch(
                        target_pos[start:end],
                        target_neg[start:end],
                        target_nz[start:end],
                        target_count[start:end],
                        train_pos_t,
                        train_neg_t,
                        train_nz_t,
                        train_count,
                    )
                )

        for weight in direction_weights:
            score = args.shape_weight * shape_norm + weight * direction_norm
            if raw_norm is not None:
                score = score + args.raw_weight * raw_norm
            if std_norm is not None:
                score = score + args.std_weight * std_norm
            if device is None:
                selected_idx, selected_score = select_topk(score, args.top_k)
            else:
                with torch.no_grad():
                    selected_score_t, selected_idx_t = torch.topk(
                        score,
                        min(args.top_k, score.shape[1]),
                        dim=1,
                        largest=False,
                        sorted=True,
                    )
                    selected_idx = selected_idx_t.cpu().numpy().astype(np.int32, copy=False)
                    selected_score = selected_score_t.cpu().numpy().astype(np.float32, copy=False)
            rerank_batches[weight].append(
                predict_selected(
                    train_windows,
                    train_preds,
                    target_windows[start:end],
                    selected_idx,
                    selected_score,
                    args,
                )
            )

    return {
        "true": target_preds,
        "baseline_pred": np.concatenate(baseline_batches, axis=0),
        "rerank_preds": {
            weight: np.concatenate(preds, axis=0)
            for weight, preds in rerank_batches.items()
        },
        "candidate_count": float(len(train_windows)),
    }


def evaluate_channels_batched(
    train_windows: np.ndarray,
    train_preds: np.ndarray,
    target_windows: np.ndarray,
    target_preds: np.ndarray,
    args,
    direction_weights: list[float],
    channel_names: list[str],
    device: torch.device,
) -> dict:
    n_channels = train_windows.shape[2]
    print(
        f"  all_channels: samples={len(target_windows)}, candidates={len(train_windows)}, "
        f"channels={n_channels}, backend={device}, channel_backend=batched",
        flush=True,
    )

    train_shape = torch_channel_shape_features(train_windows, device)
    target_shape = torch_channel_shape_features(target_windows, device)
    train_shape_sq = torch.sum(train_shape * train_shape, dim=1)

    train_raw = torch.as_tensor(train_windows, dtype=torch.float32, device=device)
    target_raw = torch.as_tensor(target_windows, dtype=torch.float32, device=device)
    train_raw_sq = torch.sum(train_raw * train_raw, dim=1)

    train_std = torch.clamp(torch.std(train_raw, dim=1, unbiased=False), min=args.std_eps)
    target_std = torch.clamp(torch.std(target_raw, dim=1, unbiased=False), min=args.std_eps)

    train_dir = torch_channel_direction_parts(train_windows, args.std_eps, device)
    target_dir = torch_channel_direction_parts(target_windows, args.std_eps, device)

    baseline_parts = []
    rerank_parts = {weight: [] for weight in direction_weights}
    for start in range(0, len(target_windows), args.selected_batch_size):
        end = min(start + args.selected_batch_size, len(target_windows))
        with torch.no_grad():
            shape_dist = torch_channel_dist_batch(target_shape[start:end], train_shape, train_shape_sq)
            baseline_score_t, baseline_idx_t = torch.topk(
                shape_dist,
                min(args.top_k, shape_dist.shape[1]),
                dim=1,
                largest=False,
                sorted=True,
            )
            baseline_idx_all = baseline_idx_t.cpu().numpy().astype(np.int32, copy=False)
            baseline_score_all = baseline_score_t.cpu().numpy().astype(np.float32, copy=False)

            shape_norm = torch_channel_row_median_normalize(shape_dist)
            direction_norm = torch_channel_row_median_normalize(
                torch_channel_direction_loss_batch(
                    {key: value[start:end] for key, value in target_dir.items()},
                    train_dir,
                )
            )

            raw_norm = None
            if args.raw_weight:
                raw_norm = torch_channel_row_median_normalize(
                    torch_channel_dist_batch(target_raw[start:end], train_raw, train_raw_sq)
                )

            std_norm = None
            if args.std_weight:
                std_norm = torch_channel_row_median_normalize(
                    torch_channel_std_loss_batch(target_std[start:end], train_std)
                )

            selected = {}
            for weight in direction_weights:
                score = args.shape_weight * shape_norm + weight * direction_norm
                if raw_norm is not None:
                    score = score + args.raw_weight * raw_norm
                if std_norm is not None:
                    score = score + args.std_weight * std_norm
                selected_score_t, selected_idx_t = torch.topk(
                    score,
                    min(args.top_k, score.shape[1]),
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                selected[weight] = (
                    selected_idx_t.cpu().numpy().astype(np.int32, copy=False),
                    selected_score_t.cpu().numpy().astype(np.float32, copy=False),
                )

        baseline_channel_preds = []
        rerank_channel_preds = {weight: [] for weight in direction_weights}
        for channel_idx in range(n_channels):
            train_windows_ch = train_windows[:, :, channel_idx:channel_idx + 1]
            train_preds_ch = train_preds[:, :, channel_idx:channel_idx + 1]
            target_windows_ch = target_windows[start:end, :, channel_idx:channel_idx + 1]

            baseline_channel_preds.append(
                predict_selected(
                    train_windows_ch,
                    train_preds_ch,
                    target_windows_ch,
                    baseline_idx_all[:, :, channel_idx],
                    baseline_score_all[:, :, channel_idx],
                    args,
                )
            )
            for weight in direction_weights:
                selected_idx_all, selected_score_all = selected[weight]
                rerank_channel_preds[weight].append(
                    predict_selected(
                        train_windows_ch,
                        train_preds_ch,
                        target_windows_ch,
                        selected_idx_all[:, :, channel_idx],
                        selected_score_all[:, :, channel_idx],
                        args,
                    )
                )

        baseline_parts.append(np.concatenate(baseline_channel_preds, axis=2))
        for weight in direction_weights:
            rerank_parts[weight].append(np.concatenate(rerank_channel_preds[weight], axis=2))

    return {
        "true": target_preds,
        "baseline_pred": np.concatenate(baseline_parts, axis=0),
        "rerank_preds": {
            weight: np.concatenate(preds, axis=0)
            for weight, preds in rerank_parts.items()
        },
        "candidate_count": float(len(train_windows)),
        "channel_count": int(n_channels),
        "channel_names": channel_names,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="dataset")
    parser.add_argument("--data_path", default="ETTm1.csv")
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--flag", default="test", choices=FLAGS)
    parser.add_argument("--weights", default="1.5", help="comma-separated same-direction weights")
    parser.add_argument("--shape_weight", type=float, default=1.0)
    parser.add_argument("--raw_weight", type=float, default=0.0)
    parser.add_argument("--std_weight", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--align_mode", default="clipped_std", choices=["none", "mean", "std", "clipped_std"])
    parser.add_argument("--std_eps", type=float, default=1e-6)
    parser.add_argument("--std_ratio_min", type=float, default=0.5)
    parser.add_argument("--std_ratio_max", type=float, default=2.0)
    parser.add_argument("--weight_mode", default="inverse", choices=["inverse", "softmax", "uniform"])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--selected_batch_size", type=int, default=128)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "cpu", "cuda", "mps"])
    parser.add_argument("--channel_backend", default="loop", choices=["loop", "batched"])
    parser.add_argument("--output_dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    direction_weights = parse_floats(args.weights)
    device = select_device(args.backend)

    train_set, train_loader = data_provider(args, "train")
    target_set, target_loader = data_provider(args, args.flag)
    train_windows_all, train_preds_all = collect_windows(train_loader, args.pred_len)
    target_windows_all, target_preds_all = collect_windows(target_loader, args.pred_len)
    key_cols = train_set.key_cols
    pred_cols = target_set.pred_cols
    if len(key_cols) != len(pred_cols):
        raise ValueError("channel-independent retrieval expects key and pred channels to match")

    data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    setting = f"ChannelIndependentDirectionStandalone_{data_name}_{args.features}_sl{args.seq_len}_pl{args.pred_len}"
    output_dir = args.output_dir or os.path.join(
        "outputs",
        "channel_independent_direction_standalone",
        f"{data_name}_sl{args.seq_len}_pl{args.pred_len}_k{args.top_k}",
    )

    print(f"setting={setting}")
    print(f"flag={args.flag}")
    print(f"top_k={args.top_k}")
    print(f"weights={list(map(fmt_float, direction_weights))}")
    print(f"backend={'numpy' if device is None else str(device)}")
    print(f"channel_backend={args.channel_backend}")
    print(
        "score="
        f"{args.shape_weight:g}*row_norm(shape_loss)"
        "+direction_weight*row_norm(same_direction_loss)"
        f"+{args.raw_weight:g}*row_norm(raw_loss)"
        f"+{args.std_weight:g}*row_norm(std_loss)"
    )

    if args.channel_backend == "batched":
        if device is None:
            raise ValueError("--channel_backend batched requires a torch backend; use --backend auto/cpu/cuda/mps")
        result = evaluate_channels_batched(
            train_windows_all,
            train_preds_all,
            target_windows_all,
            target_preds_all,
            args,
            direction_weights,
            list(pred_cols),
            device,
        )
        true = result["true"]
        baseline_pred = result["baseline_pred"]
        rerank_parts = {weight: [result["rerank_preds"][weight]] for weight in direction_weights}
        candidate_counts = [result["candidate_count"]]
    else:
        true_parts = []
        baseline_parts = []
        rerank_parts = {weight: [] for weight in direction_weights}
        candidate_counts = []
        for channel_idx, channel_name in enumerate(pred_cols):
            result = evaluate_channel(
                train_windows_all,
                train_preds_all,
                target_windows_all,
                target_preds_all,
                channel_idx,
                args,
                direction_weights,
                channel_name,
                device,
            )
            true_parts.append(result["true"])
            baseline_parts.append(result["baseline_pred"])
            for weight in direction_weights:
                rerank_parts[weight].append(result["rerank_preds"][weight])
            candidate_counts.append(result["candidate_count"])

        true = np.concatenate(true_parts, axis=2)
        baseline_pred = np.concatenate(baseline_parts, axis=2)
    baseline_metrics = metric(baseline_pred, true)

    common = {
        "data_path": args.data_path,
        "features": args.features,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "top_k": args.top_k,
        "candidate_mode": "full_stream",
        "align_mode": args.align_mode,
        "std_eps": args.std_eps,
        "std_ratio_min": args.std_ratio_min,
        "std_ratio_max": args.std_ratio_max,
        "direction_loss": "same_direction",
        "weight_mode": args.weight_mode,
        "backend": "numpy" if device is None else str(device),
        "channel_backend": args.channel_backend,
        "channel_count": len(pred_cols),
        "candidate_mean": float(np.mean(candidate_counts)),
        "candidate_mean_min": float(np.min(candidate_counts)),
        "candidate_mean_max": float(np.max(candidate_counts)),
        "shape_weight": args.shape_weight,
        "raw_weight": args.raw_weight,
        "std_weight": args.std_weight,
        "baseline_mse": baseline_metrics[1],
        "baseline_mae": baseline_metrics[0],
    }

    rows = [
        metric_row(
            setting,
            args.flag,
            "shape_baseline_ci",
            baseline_metrics,
            {
                **common,
                "direction_weight": 0.0,
                "delta_mse_vs_shape": 0.0,
                "delta_mae_vs_shape": 0.0,
            },
        )
    ]

    print("\n| weight | test MSE | test MAE | delta MSE vs shape |")
    print("|---:|---:|---:|---:|")
    print(f"| baseline | {baseline_metrics[1]:.6f} | {baseline_metrics[0]:.6f} | 0.000000 |")
    for weight in direction_weights:
        rerank_pred = np.concatenate(rerank_parts[weight], axis=2)
        rerank_metrics = metric(rerank_pred, true)
        row = metric_row(
            setting,
            args.flag,
            "rerank_ci",
            rerank_metrics,
            {
                **common,
                "direction_weight": weight,
                "delta_mse_vs_shape": baseline_metrics[1] - rerank_metrics[1],
                "delta_mae_vs_shape": baseline_metrics[0] - rerank_metrics[0],
            },
        )
        rows.append(row)
        print(
            f"| {fmt_float(weight)} | {row['mse']:.6f} | {row['mae']:.6f} | "
            f"{row['delta_mse_vs_shape']:.6f} |"
        )

    summary_path = Path(output_dir) / "summary.csv"
    write_rows(summary_path, rows)
    print(f"\nsummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
