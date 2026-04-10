import numpy as np

from retrieval.distance import l2_distance_matrix


def exact_l2_rerank(
    query_flat: np.ndarray,
    memory_flat: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    b, c = candidate_ids.shape
    out_ids = np.zeros((b, min(top_k, c)), dtype=np.int64)
    out_scores = np.zeros((b, min(top_k, c)), dtype=np.float32)
    for i in range(b):
        ids = candidate_ids[i]
        cand = memory_flat[ids]
        d = l2_distance_matrix(query_flat[i:i + 1], cand)[0]
        k = min(top_k, len(ids))
        order = np.argsort(d)[:k]
        out_ids[i, :k] = ids[order]
        out_scores[i, :k] = d[order]
    return out_ids, out_scores
