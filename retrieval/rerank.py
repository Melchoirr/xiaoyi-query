import numpy as np

from retrieval.distance import l2_distance_matrix


def exact_l2_rerank(query_flat: np.ndarray, memory_flat: np.ndarray, candidate_ids: np.ndarray, top_k: int):
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


def hybrid_rerank(
    candidate_ids: np.ndarray,
    shape_scores: np.ndarray,
    query_mean: np.ndarray,
    query_std: np.ndarray,
    query_last: np.ndarray,
    memory_mean: np.ndarray,
    memory_std: np.ndarray,
    memory_last: np.ndarray,
    query_phase: np.ndarray | None,
    memory_phase: np.ndarray | None,
    top_k: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
):
    b, c = candidate_ids.shape
    out_ids = np.zeros((b, min(top_k, c)), dtype=np.int64)
    out_scores = np.zeros((b, min(top_k, c)), dtype=np.float32)

    decomp = {
        "shape": np.zeros((b, c), dtype=np.float32),
        "level": np.zeros((b, c), dtype=np.float32),
        "scale": np.zeros((b, c), dtype=np.float32),
        "phase": np.zeros((b, c), dtype=np.float32),
        "total": np.zeros((b, c), dtype=np.float32),
    }

    for i in range(b):
        ids = candidate_ids[i]
        s = shape_scores[i]
        l = np.linalg.norm(memory_mean[ids] - query_mean[i:i + 1], axis=1)
        sc = np.linalg.norm(memory_std[ids] - query_std[i:i + 1], axis=1)
        lv = np.linalg.norm(memory_last[ids] - query_last[i:i + 1], axis=1)
        level = 0.5 * l + 0.5 * lv

        if query_phase is not None and memory_phase is not None:
            p = np.linalg.norm(memory_phase[ids] - query_phase[i:i + 1], axis=1)
        else:
            p = np.zeros_like(s)

        total = alpha * s + beta * level + gamma * sc + delta * p
        k = min(top_k, len(ids))
        order = np.argsort(total)[:k]

        out_ids[i, :k] = ids[order]
        out_scores[i, :k] = total[order].astype(np.float32)

        decomp["shape"][i] = s
        decomp["level"][i] = level
        decomp["scale"][i] = sc
        decomp["phase"][i] = p
        decomp["total"][i] = total

    return out_ids, out_scores, decomp
