import numpy as np


def aggregate_futures(candidate_futures: np.ndarray, scores: np.ndarray, mode: str = "inverse_distance") -> np.ndarray:
    if candidate_futures.ndim != 4:
        raise ValueError("candidate_futures must be [B, K, pred_len, C]")
    if scores.ndim != 2:
        raise ValueError("scores must be [B, K]")

    if mode == "mean":
        w = np.full(scores.shape, 1.0 / scores.shape[1], dtype=np.float32)
    elif mode == "softmax":
        logits = -scores
        logits = logits - logits.max(axis=1, keepdims=True)
        expv = np.exp(logits)
        w = expv / np.clip(expv.sum(axis=1, keepdims=True), 1e-8, None)
    elif mode == "inverse_distance":
        inv = 1.0 / np.clip(scores, 1e-8, None)
        w = inv / np.clip(inv.sum(axis=1, keepdims=True), 1e-8, None)
    else:
        raise ValueError(f"unknown aggregation mode: {mode}")

    return (candidate_futures * w[:, :, None, None]).sum(axis=1)
