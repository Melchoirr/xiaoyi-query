"""
Retrieval-based time-series forecasting baseline.

Builds a train retrieval bank from historical value windows plus aligned time
feature windows. For each query window from train/val/test, it retrieves Top-K
train windows per target channel and returns the weighted average of their
future windows.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from forecast.utils.metrics import metric
from forecast.utils.timefeatures import time_features


FLAGS = ("train", "val", "test")


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ett_borders(data_name, data_path, seq_len):
    name = (data_name or os.path.splitext(os.path.basename(data_path))[0]).lower()
    if name.startswith("ettm"):
        unit = 4
    else:
        unit = 1

    train = 12 * 30 * 24 * unit
    val = 4 * 30 * 24 * unit
    test = 4 * 30 * 24 * unit
    border1s = [0, train - seq_len, train + val - seq_len]
    border2s = [train, train + val, train + val + test]
    return border1s, border2s


def _select_columns(df_raw, features, target):
    value_cols = list(df_raw.columns[1:])
    if features == "M":
        key_cols = value_cols
        pred_cols = value_cols
    elif features == "S":
        if target not in value_cols:
            raise ValueError(f"target column {target!r} not found in data")
        key_cols = [target]
        pred_cols = [target]
    elif features == "MS":
        if target not in value_cols:
            raise ValueError(f"target column {target!r} not found in data")
        key_cols = value_cols
        pred_cols = [target]
    else:
        raise ValueError(f"unsupported features mode: {features}")
    return key_cols, pred_cols


def load_ett_data(root_path, data_path, data_name, features, target, seq_len, freq):
    """Load ETT data and return split arrays for retrieval."""
    csv_path = os.path.join(root_path, data_path)
    df_raw = pd.read_csv(csv_path)
    border1s, border2s = _ett_borders(data_name, data_path, seq_len)
    key_cols, pred_cols = _select_columns(df_raw, features, target)

    raw_key = df_raw[key_cols].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(raw_key[border1s[0]:border2s[0]])
    norm_key = scaler.transform(raw_key).astype(np.float32)

    pred_positions = [key_cols.index(col) for col in pred_cols]
    norm_pred = norm_key[:, pred_positions].astype(np.float32)

    dates = pd.to_datetime(df_raw["date"].values)
    time_data = time_features(dates, freq=freq).transpose(1, 0).astype(np.float32)

    splits = {}
    for i, flag in enumerate(FLAGS):
        border1, border2 = border1s[i], border2s[i]
        splits[flag] = {
            "key": norm_key[border1:border2],
            "pred": norm_pred[border1:border2],
            "time": time_data[border1:border2],
        }

    return splits, key_cols, pred_cols


def build_windows(key_data, pred_data, time_data, seq_len, pred_len):
    """Build value, target, and time windows from one continuous split."""
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


def _squared_distance_matrix(query, bank, bank_sq, denom):
    query_sq = (query ** 2).sum(dim=1, keepdim=True)
    dists = query_sq + bank_sq.unsqueeze(0) - 2.0 * (query @ bank.T)
    return torch.clamp(dists / denom, min=0.0)


def retrieval_match(
    target_keys,
    train_keys,
    train_preds,
    target_time_keys,
    train_time_keys,
    exclude_self=False,
    top_k=5,
    batch_size=512,
    value_weight=1.0,
    time_weight=1.0,
):
    """Retrieve Top-K train futures for each query window.

    If the number of key channels equals the number of prediction channels, the
    value distance is computed independently per channel. This is the default M
    and S behavior. If they differ, the value key is flattened across all input
    channels and reused for each prediction channel; this supports MS mode.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if value_weight < 0 or time_weight < 0:
        raise ValueError("value_weight and time_weight must be non-negative")
    if value_weight == 0 and time_weight == 0:
        raise ValueError("at least one distance weight must be positive")

    device = _get_device()
    n_target, seq_len, key_channels = target_keys.shape
    n_train = train_keys.shape[0]
    pred_len = train_preds.shape[1]
    pred_channels = train_preds.shape[2]
    effective_top_k = min(top_k, n_train - 1 if exclude_self else n_train)
    if effective_top_k <= 0:
        raise ValueError("retrieval bank is empty after self-exclusion")

    matched_preds = np.zeros((n_target, pred_len, pred_channels), dtype=np.float32)
    all_topk_idx = np.zeros((pred_channels, n_target, effective_top_k), dtype=np.int32)
    all_topk_dist = np.zeros((pred_channels, n_target, effective_top_k), dtype=np.float32)
    all_topk_weight = np.zeros((pred_channels, n_target, effective_top_k), dtype=np.float32)

    train_time = torch.from_numpy(train_time_keys.reshape(n_train, -1)).float().to(device)
    train_time_sq = (train_time ** 2).sum(dim=1)
    time_denom = max(1, train_time.shape[1])

    use_per_channel_value = key_channels == pred_channels
    train_value_flat = None
    train_value_flat_sq = None
    value_flat_denom = None
    train_value_tensors = None
    train_value_sqs = None
    if not use_per_channel_value:
        train_value_flat = torch.from_numpy(train_keys.reshape(n_train, -1)).float().to(device)
        train_value_flat_sq = (train_value_flat ** 2).sum(dim=1)
        value_flat_denom = max(1, train_value_flat.shape[1])
    else:
        train_value_tensors = [
            torch.from_numpy(train_keys[:, :, d]).float().to(device)
            for d in range(pred_channels)
        ]
        train_value_sqs = [(train_value ** 2).sum(dim=1) for train_value in train_value_tensors]

    train_pred_tensors = [
        torch.from_numpy(train_preds[:, :, d]).float().to(device)
        for d in range(pred_channels)
    ]

    for start in range(0, n_target, batch_size):
        end = min(start + batch_size, n_target)
        print(f"  {start}/{n_target}")

        query_time = torch.from_numpy(target_time_keys[start:end].reshape(end - start, -1)).float().to(device)
        time_dists = _squared_distance_matrix(
            query_time,
            train_time,
            train_time_sq,
            time_denom,
        )

        value_dists_flat = None
        if not use_per_channel_value:
            query_value_flat = torch.from_numpy(target_keys[start:end].reshape(end - start, -1)).float().to(device)
            value_dists_flat = _squared_distance_matrix(
                query_value_flat,
                train_value_flat,
                train_value_flat_sq,
                value_flat_denom,
            )

        for d in range(pred_channels):
            if use_per_channel_value:
                train_value = train_value_tensors[d]
                train_value_sq = train_value_sqs[d]
                query_value = torch.from_numpy(target_keys[start:end, :, d]).float().to(device)
                value_dists = _squared_distance_matrix(
                    query_value,
                    train_value,
                    train_value_sq,
                    seq_len,
                )
            else:
                value_dists = value_dists_flat

            dists = value_weight * value_dists + time_weight * time_dists
            if exclude_self:
                idx = torch.arange(start, end, device=device)
                mask = idx < n_train
                if mask.any():
                    dists[mask, idx[mask]] = float("inf")

            topk_dist, topk_idx = torch.topk(
                dists,
                effective_top_k,
                dim=1,
                largest=False,
            )
            weights = 1.0 / (topk_dist + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)

            train_pred = train_pred_tensors[d]
            gathered = train_pred[topk_idx.reshape(-1)].reshape(
                end - start,
                effective_top_k,
                pred_len,
            )
            weighted = (weights.unsqueeze(-1) * gathered).sum(dim=1)

            matched_preds[start:end, :, d] = weighted.cpu().numpy()
            all_topk_idx[d, start:end] = topk_idx.cpu().numpy().astype(np.int32)
            all_topk_dist[d, start:end] = topk_dist.cpu().numpy().astype(np.float32)
            all_topk_weight[d, start:end] = weights.cpu().numpy().astype(np.float32)

    details = {
        "topk_idx": all_topk_idx,
        "topk_dist": all_topk_dist,
        "topk_weight": all_topk_weight,
    }
    return matched_preds, details


def _parse_flags(flags):
    parsed = [flag.strip() for flag in flags.split(",") if flag.strip()]
    invalid = [flag for flag in parsed if flag not in FLAGS]
    if invalid:
        raise ValueError(f"invalid flags: {invalid}; expected any of {FLAGS}")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="dataset")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--flags", type=str, default="test,train,val")
    parser.add_argument("--match_top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--value_weight", type=float, default=1.0)
    parser.add_argument("--time_weight", type=float, default=1.0)
    parser.add_argument("--result_path", type=str, default="outputs/results")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    setting = f"RetrievalMatch_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}"
    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_path, setting)
    os.makedirs(args.output_dir, exist_ok=True)

    flags = _parse_flags(args.flags)
    print(f"RetrievalMatch setting: {setting}")
    print(f"flags={flags}, top_k={args.match_top_k}, value_weight={args.value_weight}, time_weight={args.time_weight}")

    splits, key_cols, pred_cols = load_ett_data(
        args.root_path,
        args.data_path,
        args.data,
        args.features,
        args.target,
        args.seq_len,
        args.freq,
    )
    print(f"key columns: {key_cols}")
    print(f"prediction columns: {pred_cols}")

    train_keys, train_preds, train_time_keys = build_windows(
        splits["train"]["key"],
        splits["train"]["pred"],
        splits["train"]["time"],
        args.seq_len,
        args.pred_len,
    )
    print(f"train retrieval bank: {len(train_keys)}")

    for flag in flags:
        print(f"\n{'=' * 40}")
        print(f"processing {flag} split")

        target_keys, target_preds, target_time_keys = build_windows(
            splits[flag]["key"],
            splits[flag]["pred"],
            splits[flag]["time"],
            args.seq_len,
            args.pred_len,
        )
        print(f"{flag} samples: {len(target_keys)}")

        matched_preds, details = retrieval_match(
            target_keys,
            train_keys,
            train_preds,
            target_time_keys,
            train_time_keys,
            exclude_self=(flag == "train"),
            top_k=args.match_top_k,
            batch_size=args.batch_size,
            value_weight=args.value_weight,
            time_weight=args.time_weight,
        )

        mae, mse, rmse, mape, mspe = metric(matched_preds, target_preds)
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

        if flag == "test":
            save_dir = args.output_dir
        else:
            save_dir = os.path.join(args.output_dir, flag)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "pred.npy"), matched_preds)
        np.save(os.path.join(save_dir, "true.npy"), target_preds)
        np.save(os.path.join(save_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(save_dir, "topk_idx.npy"), details["topk_idx"])
        np.save(os.path.join(save_dir, "topk_dist.npy"), details["topk_dist"])
        np.save(os.path.join(save_dir, "topk_weight.npy"), details["topk_weight"])
        print(f"  saved to: {save_dir}")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"RESULT|{ts}|{setting}|{flag}|"
            f"mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}"
        )

    print("Done!")


if __name__ == "__main__":
    main()
