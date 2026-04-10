from typing import Tuple

import numpy as np


def flatten_windows(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("input must be [N, L, C]")
    return x.reshape(x.shape[0], -1)


def window_z_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.clip(std, eps, None)
    return (x - mean) / std


def l2_distance_matrix(query_flat: np.ndarray, memory_flat: np.ndarray) -> np.ndarray:
    q2 = (query_flat ** 2).sum(axis=1, keepdims=True)
    m2 = (memory_flat ** 2).sum(axis=1, keepdims=True).T
    cross = query_flat @ memory_flat.T
    dist2 = np.maximum(q2 + m2 - 2.0 * cross, 0.0)
    return np.sqrt(dist2).astype(np.float32)


def cosine_similarity_matrix(query_vecs: np.ndarray, memory_vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    qn = query_vecs / np.clip(np.linalg.norm(query_vecs, axis=1, keepdims=True), eps, None)
    mn = memory_vecs / np.clip(np.linalg.norm(memory_vecs, axis=1, keepdims=True), eps, None)
    return qn @ mn.T


def dtw_distance(query: np.ndarray, candidate: np.ndarray, radius: int = 5) -> float:
    l = query.shape[0]
    r = max(1, min(radius, l))
    dp = np.full((l + 1, l + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, l + 1):
        j_start = max(1, i - r)
        j_end = min(l, i + r)
        for j in range(j_start, j_end + 1):
            cost = np.linalg.norm(query[i - 1] - candidate[j - 1]).astype(np.float32)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[l, l])
