"""
Embedding retrieval + reranking baseline for ETTh1 forecasting.

This script is intentionally standalone and does not modify forecast/run.py.
It builds lightweight window embeddings, retrieves candidates from the train
bank, reranks them with multiple time-series distances, and predicts by a
softmax-weighted average of the selected train future windows.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecast.utils.metrics import metric
from forecast.utils.timefeatures import time_features


FLAGS = ("train", "val", "test")
RERANK_WEIGHTS = {
    "embed": 0.25,
    "value": 0.40,
    "diff": 0.20,
    "time": 0.10,
    "last": 0.05,
}
FFT_BINS = 4
MAX_RERANK_CHUNK_ELEMENTS = 16_000_000


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _data_name_from_path(data_path):
    return os.path.splitext(os.path.basename(data_path))[0]


def _ett_borders(data_path, seq_len):
    name = _data_name_from_path(data_path).lower()
    unit = 4 if name.startswith("ettm") else 1
    train = 12 * 30 * 24 * unit
    val = 4 * 30 * 24 * unit
    test = 4 * 30 * 24 * unit
    border1s = [0, train - seq_len, train + val - seq_len]
    border2s = [train, train + val, train + val + test]
    return border1s, border2s


def _select_columns(df_raw, features, target):
    value_cols = list(df_raw.columns[1:])
    if target not in value_cols:
        raise ValueError(f"target column {target!r} not found in data")

    if features == "M":
        key_cols = value_cols
        pred_cols = value_cols
    elif features == "S":
        key_cols = [target]
        pred_cols = [target]
    elif features == "MS":
        key_cols = value_cols
        pred_cols = [target]
    else:
        raise ValueError(f"unsupported features mode: {features}")
    return value_cols, key_cols, pred_cols


def load_ett_data(root_path, data_path, features, target, seq_len, freq):
    csv_path = os.path.join(root_path, data_path)
    df_raw = pd.read_csv(csv_path)
    if "date" not in df_raw.columns:
        raise ValueError(f"{csv_path} must contain a 'date' column")

    border1s, border2s = _ett_borders(data_path, seq_len)
    value_cols, key_cols, pred_cols = _select_columns(df_raw, features, target)

    raw_values = df_raw[value_cols].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(raw_values[border1s[0]:border2s[0]])
    norm_values = scaler.transform(raw_values).astype(np.float32)

    key_positions = [value_cols.index(col) for col in key_cols]
    pred_positions = [value_cols.index(col) for col in pred_cols]
    norm_key = norm_values[:, key_positions].astype(np.float32)
    norm_pred = norm_values[:, pred_positions].astype(np.float32)

    dates = pd.to_datetime(df_raw["date"].values)
    time_data = time_features(dates, freq=freq).transpose(1, 0).astype(np.float32)

    splits = {}
    for i, flag in enumerate(FLAGS):
        border1, border2 = border1s[i], border2s[i]
        if border1 < 0 or border2 > len(df_raw):
            raise ValueError(
                f"invalid split borders for {data_path}: {border1=}, {border2=}, "
                f"data length={len(df_raw)}"
            )
        splits[flag] = {
            "key": norm_key[border1:border2],
            "pred": norm_pred[border1:border2],
            "time": time_data[border1:border2],
            "border": (border1, border2),
        }

    return splits, key_cols, pred_cols


def build_windows(key_data, pred_data, time_data, seq_len, pred_len):
    n_samples = len(key_data) - seq_len - pred_len + 1
    if n_samples <= 0:
        raise ValueError(
            f"not enough data for seq_len={seq_len}, pred_len={pred_len}: "
            f"split length={len(key_data)}"
        )

    value_windows = np.asarray(
        [key_data[i:i + seq_len] for i in range(n_samples)],
        dtype=np.float32,
    )
    pred_windows = np.asarray(
        [pred_data[i + seq_len:i + seq_len + pred_len] for i in range(n_samples)],
        dtype=np.float32,
    )
    time_windows = np.asarray(
        [time_data[i:i + seq_len] for i in range(n_samples)],
        dtype=np.float32,
    )
    return value_windows, pred_windows, time_windows


def build_embeddings(value_windows, time_windows, fft_bins=FFT_BINS):
    n_samples, seq_len, n_channels = value_windows.shape
    parts = [
        value_windows.mean(axis=1),
        value_windows.std(axis=1),
        value_windows[:, -1, :] - value_windows[:, 0, :],
        value_windows[:, -1, :],
    ]

    if seq_len > 1:
        x = np.arange(seq_len, dtype=np.float32)
        x = x - x.mean()
        denom = float(np.sum(x * x))
        slopes = np.einsum("nld,l->nd", value_windows, x, optimize=True) / denom
        diffs = np.diff(value_windows, axis=1)
        parts.extend([slopes.astype(np.float32), diffs.mean(axis=1), diffs.std(axis=1)])
    else:
        zeros = np.zeros((n_samples, n_channels), dtype=np.float32)
        parts.extend([zeros, zeros, zeros])

    fft_values = np.fft.rfft(value_windows, axis=1)
    available_bins = max(0, min(fft_bins, fft_values.shape[1] - 1))
    fft_amp = np.zeros((n_samples, fft_bins, n_channels), dtype=np.float32)
    if available_bins > 0:
        fft_amp[:, :available_bins, :] = (
            np.abs(fft_values[:, 1:available_bins + 1, :]).astype(np.float32)
            / max(1, seq_len)
        )
    parts.append(fft_amp.reshape(n_samples, -1))

    if time_windows.shape[2] > 0:
        parts.extend([time_windows.mean(axis=1), time_windows.std(axis=1)])

    embedding = np.concatenate([part.reshape(n_samples, -1) for part in parts], axis=1)
    embedding = embedding.astype(np.float32, copy=False)
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norm, 1e-8)
    return embedding.astype(np.float32, copy=False)


def _flatten(array):
    return np.ascontiguousarray(array.reshape(array.shape[0], -1), dtype=np.float32)


def _squared_norm(flat):
    if flat.shape[1] == 0:
        return np.zeros((flat.shape[0],), dtype=np.float32)
    return np.einsum("nf,nf->n", flat, flat, optimize=True).astype(np.float32)


def build_distance_cache(value_windows, time_windows):
    diffs = np.diff(value_windows, axis=1)
    cache = {
        "value": _flatten(value_windows),
        "diff": _flatten(diffs),
        "time": _flatten(time_windows),
        "last": np.ascontiguousarray(value_windows[:, -1, :], dtype=np.float32),
    }
    for key in ("value", "diff", "time", "last"):
        cache[f"{key}_sq"] = _squared_norm(cache[key])
        cache[f"{key}_denom"] = max(1, cache[key].shape[1])
    return cache


def _selected_mse(query_flat, train_flat, train_sq, train_idx, denom):
    if query_flat.shape[1] == 0:
        return np.zeros(train_idx.shape, dtype=np.float32)

    batch, n_candidates = train_idx.shape
    out = np.empty((batch, n_candidates), dtype=np.float32)
    query_sq = _squared_norm(query_flat)
    feature_dim = query_flat.shape[1]
    chunk = max(1, MAX_RERANK_CHUNK_ELEMENTS // max(1, batch * feature_dim))
    chunk = min(n_candidates, chunk)

    for start in range(0, n_candidates, chunk):
        end = min(start + chunk, n_candidates)
        idx = train_idx[:, start:end]
        train_chunk = train_flat[idx]
        dot = np.einsum("bf,bcf->bc", query_flat, train_chunk, optimize=True)
        dist = query_sq[:, None] + train_sq[idx] - 2.0 * dot
        out[:, start:end] = np.maximum(dist / float(denom), 0.0)

    return out


def _softmax_negative_distance(distances, temperature):
    logits = -distances / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits).astype(np.float32)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32, copy=False)


def _channel_compatible_topk(array, pred_channels):
    return np.broadcast_to(array[None, ...], (pred_channels,) + array.shape).copy()


def embed_retrieval_rerank(
    target_values,
    target_preds,
    target_times,
    target_embeddings,
    train_values,
    train_preds,
    train_times,
    train_embeddings,
    candidate_k,
    rerank_k,
    batch_size,
    temperature,
    exclude_self=False,
    train_cache=None,
    target_cache=None,
):
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if rerank_k <= 0:
        raise ValueError("rerank_k must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    device = _get_device()
    n_target = target_values.shape[0]
    n_train = train_values.shape[0]
    pred_len = train_preds.shape[1]
    pred_channels = train_preds.shape[2]
    effective_candidate_k = min(candidate_k, n_train - 1 if exclude_self else n_train)
    if effective_candidate_k <= 0:
        raise ValueError("retrieval bank is empty after self-exclusion")
    effective_rerank_k = min(rerank_k, effective_candidate_k)

    matched_preds = np.zeros((n_target, pred_len, pred_channels), dtype=np.float32)
    topk_idx = np.zeros((n_target, effective_rerank_k), dtype=np.int32)
    topk_dist = np.zeros((n_target, effective_rerank_k), dtype=np.float32)
    topk_weight = np.zeros((n_target, effective_rerank_k), dtype=np.float32)

    if train_cache is None:
        train_cache = build_distance_cache(train_values, train_times)
    if target_cache is None:
        if target_values is train_values and target_times is train_times:
            target_cache = train_cache
        else:
            target_cache = build_distance_cache(target_values, target_times)
    train_embedding_tensor = torch.from_numpy(train_embeddings).float().to(device)

    for start in range(0, n_target, batch_size):
        end = min(start + batch_size, n_target)
        print(f"  {start}/{n_target}")

        query_embeddings = torch.from_numpy(target_embeddings[start:end]).float().to(device)
        similarities = query_embeddings @ train_embedding_tensor.T
        if exclude_self:
            self_idx = torch.arange(start, end, device=device)
            mask = self_idx < n_train
            if mask.any():
                similarities[mask, self_idx[mask]] = -float("inf")

        candidate_sim, candidate_idx_t = torch.topk(
            similarities,
            effective_candidate_k,
            dim=1,
            largest=True,
        )
        candidate_idx = candidate_idx_t.cpu().numpy().astype(np.int64)
        embed_dist = np.maximum(1.0 - candidate_sim.cpu().numpy(), 0.0).astype(np.float32)

        value_dist = _selected_mse(
            target_cache["value"][start:end],
            train_cache["value"],
            train_cache["value_sq"],
            candidate_idx,
            train_cache["value_denom"],
        )
        diff_dist = _selected_mse(
            target_cache["diff"][start:end],
            train_cache["diff"],
            train_cache["diff_sq"],
            candidate_idx,
            train_cache["diff_denom"],
        )
        time_dist = _selected_mse(
            target_cache["time"][start:end],
            train_cache["time"],
            train_cache["time_sq"],
            candidate_idx,
            train_cache["time_denom"],
        )
        last_dist = _selected_mse(
            target_cache["last"][start:end],
            train_cache["last"],
            train_cache["last_sq"],
            candidate_idx,
            train_cache["last_denom"],
        )

        combined = (
            RERANK_WEIGHTS["embed"] * embed_dist
            + RERANK_WEIGHTS["value"] * value_dist
            + RERANK_WEIGHTS["diff"] * diff_dist
            + RERANK_WEIGHTS["time"] * time_dist
            + RERANK_WEIGHTS["last"] * last_dist
        ).astype(np.float32, copy=False)

        partial = np.argpartition(combined, effective_rerank_k - 1, axis=1)[:, :effective_rerank_k]
        partial_dist = np.take_along_axis(combined, partial, axis=1)
        order = np.argsort(partial_dist, axis=1)
        rerank_pos = np.take_along_axis(partial, order, axis=1)
        batch_idx = np.take_along_axis(candidate_idx, rerank_pos, axis=1).astype(np.int32)
        batch_dist = np.take_along_axis(combined, rerank_pos, axis=1).astype(np.float32)
        batch_weight = _softmax_negative_distance(batch_dist, temperature)

        futures = train_preds[batch_idx]
        matched_preds[start:end] = np.einsum(
            "br,brpc->bpc",
            batch_weight,
            futures,
            optimize=True,
        ).astype(np.float32)

        topk_idx[start:end] = batch_idx
        topk_dist[start:end] = batch_dist
        topk_weight[start:end] = batch_weight

    if exclude_self:
        self_hits = topk_idx == np.arange(n_target, dtype=np.int32)[:, None]
        if np.any(self_hits):
            raise RuntimeError("leave-one-out failed: train top-k contains self index")

    details = {
        "topk_idx": _channel_compatible_topk(topk_idx, pred_channels),
        "topk_dist": _channel_compatible_topk(topk_dist, pred_channels),
        "topk_weight": _channel_compatible_topk(topk_weight, pred_channels),
    }
    return matched_preds, details, effective_candidate_k, effective_rerank_k


def _parse_flags(flags):
    parsed = [flag.strip() for flag in flags.split(",") if flag.strip()]
    invalid = [flag for flag in parsed if flag not in FLAGS]
    if invalid:
        raise ValueError(f"invalid flags: {invalid}; expected any of {FLAGS}")
    if not parsed:
        raise ValueError("at least one flag must be provided")
    return parsed


def _save_results(save_dir, matched_preds, target_preds, details, metrics):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "pred.npy"), matched_preds)
    np.save(os.path.join(save_dir, "true.npy"), target_preds)
    np.save(os.path.join(save_dir, "metrics.npy"), np.array(metrics, dtype=np.float32))
    np.save(os.path.join(save_dir, "topk_idx.npy"), details["topk_idx"])
    np.save(os.path.join(save_dir, "topk_dist.npy"), details["topk_dist"])
    np.save(os.path.join(save_dir, "topk_weight.npy"), details["topk_weight"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="dataset")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--flags", type=str, default="test,train,val")
    parser.add_argument("--candidate_k", type=int, default=400)
    parser.add_argument("--rerank_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    data_name = _data_name_from_path(args.data_path)
    setting = (
        f"EmbedRetrievalRerank_{data_name}_{args.features}"
        f"_sl{args.seq_len}_pl{args.pred_len}"
    )
    output_dir = args.output_dir or os.path.join("outputs", "results", setting)
    flags = _parse_flags(args.flags)

    print(f"EmbedRetrievalRerank setting: {setting}")
    print(
        f"flags={flags}, candidate_k={args.candidate_k}, rerank_k={args.rerank_k}, "
        f"batch_size={args.batch_size}, temperature={args.temperature}"
    )
    print(f"rerank weights: {RERANK_WEIGHTS}")

    splits, key_cols, pred_cols = load_ett_data(
        args.root_path,
        args.data_path,
        args.features,
        args.target,
        args.seq_len,
        args.freq,
    )
    print(f"key columns: {key_cols}")
    print(f"prediction columns: {pred_cols}")

    print("building train retrieval bank")
    train_values, train_preds, train_times = build_windows(
        splits["train"]["key"],
        splits["train"]["pred"],
        splits["train"]["time"],
        args.seq_len,
        args.pred_len,
    )
    train_embeddings = build_embeddings(train_values, train_times)
    train_cache = build_distance_cache(train_values, train_times)
    print(f"train samples: {len(train_values)}, embedding dim: {train_embeddings.shape[1]}")

    for flag in flags:
        print(f"\n{'=' * 40}")
        print(f"processing {flag} split")

        if flag == "train":
            target_values = train_values
            target_preds = train_preds
            target_times = train_times
            target_embeddings = train_embeddings
            target_cache = train_cache
        else:
            target_values, target_preds, target_times = build_windows(
                splits[flag]["key"],
                splits[flag]["pred"],
                splits[flag]["time"],
                args.seq_len,
                args.pred_len,
            )
            target_embeddings = build_embeddings(target_values, target_times)
            target_cache = build_distance_cache(target_values, target_times)

        print(f"{flag} samples: {len(target_values)}")
        matched_preds, details, effective_candidate_k, effective_rerank_k = embed_retrieval_rerank(
            target_values=target_values,
            target_preds=target_preds,
            target_times=target_times,
            target_embeddings=target_embeddings,
            train_values=train_values,
            train_preds=train_preds,
            train_times=train_times,
            train_embeddings=train_embeddings,
            candidate_k=args.candidate_k,
            rerank_k=args.rerank_k,
            batch_size=args.batch_size,
            temperature=args.temperature,
            exclude_self=(flag == "train"),
            train_cache=train_cache,
            target_cache=target_cache,
        )

        mae, mse, rmse, mape, mspe = metric(matched_preds, target_preds)
        print(
            f"  effective_candidate_k={effective_candidate_k}, "
            f"effective_rerank_k={effective_rerank_k}"
        )
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

        save_dir = output_dir if flag == "test" else os.path.join(output_dir, flag)
        _save_results(save_dir, matched_preds, target_preds, details, (mae, mse, rmse, mape, mspe))
        print(f"  saved to: {save_dir}")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"RESULT|{ts}|{setting}|{flag}|"
            f"mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}"
        )

    print("Done!")


if __name__ == "__main__":
    main()
