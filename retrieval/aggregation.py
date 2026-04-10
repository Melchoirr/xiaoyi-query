import numpy as np


def compute_weights(scores: np.ndarray, mode: str = "inverse_distance", temperature: float = 1.0) -> np.ndarray:
    if mode == "mean":
        w = np.full(scores.shape, 1.0 / scores.shape[1], dtype=np.float32)
    elif mode == "top1":
        w = np.zeros_like(scores, dtype=np.float32)
        top = np.argmin(scores, axis=1)
        w[np.arange(scores.shape[0]), top] = 1.0
    elif mode == "inverse_distance":
        inv = 1.0 / np.clip(scores, 1e-8, None)
        w = inv / np.clip(inv.sum(axis=1, keepdims=True), 1e-8, None)
    elif mode == "softmax_temp":
        temp = max(float(temperature), 1e-6)
        logits = -scores / temp
        logits = logits - logits.max(axis=1, keepdims=True)
        expv = np.exp(logits)
        w = expv / np.clip(expv.sum(axis=1, keepdims=True), 1e-8, None)
    elif mode == "rank_based":
        rank = np.argsort(np.argsort(scores, axis=1), axis=1) + 1
        inv_rank = 1.0 / rank.astype(np.float32)
        w = inv_rank / inv_rank.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown aggregation mode: {mode}")
    return w.astype(np.float32)


def aggregate_futures(candidate_futures: np.ndarray, scores: np.ndarray, mode: str = "inverse_distance", temperature: float = 1.0):
    if candidate_futures.ndim != 4:
        raise ValueError("candidate_futures must be [B, K, pred_len, C]")
    if scores.ndim != 2:
        raise ValueError("scores must be [B, K]")

    w = compute_weights(scores, mode=mode, temperature=temperature)
    pred = (candidate_futures * w[:, :, None, None]).sum(axis=1)

    entropy = -(w * np.log(np.clip(w, 1e-12, None))).sum(axis=1)
    stats = {
        "weight_entropy_mean": float(np.mean(entropy)),
        "weight_max_mean": float(np.mean(np.max(w, axis=1))),
        "weight_min_mean": float(np.mean(np.min(w, axis=1))),
    }
    return pred.astype(np.float32), w.astype(np.float32), stats
