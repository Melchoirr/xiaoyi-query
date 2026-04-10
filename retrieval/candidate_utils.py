import numpy as np


def topk_from_distances(distances: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if distances.ndim != 2:
        raise ValueError("distances must be [B, N]")
    k = min(top_k, distances.shape[1])
    idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(distances, idx, axis=1)
    order = np.argsort(vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return idx.astype(np.int64), vals.astype(np.float32)


def pad_candidates(candidate_ids: np.ndarray, required_k: int, fallback_id: int = 0) -> np.ndarray:
    if candidate_ids.shape[1] >= required_k:
        return candidate_ids[:, :required_k]
    pad = np.full((candidate_ids.shape[0], required_k - candidate_ids.shape[1]), fallback_id, dtype=np.int64)
    return np.concatenate([candidate_ids, pad], axis=1)
