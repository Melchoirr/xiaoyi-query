from __future__ import annotations

from collections import defaultdict

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import flatten_windows, l2_distance_matrix
from retrieval.rerank import exact_l2_rerank


class LSHSearch(BaseRetrieverForecaster):
    """Approximate recall + exact rerank forecaster.

    Role: candidate recall accelerator, then exact rerank.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
        n_hash_funcs: int = 16,
        n_tables: int = 4,
        recall_k: int = 64,
        normalization: str = "window_zscore",
        random_state: int = 42,
    ) -> None:
        super().__init__(seq_len, pred_len, top_k, normalization=normalization, aggregation="inverse_distance")
        self.n_hash_funcs = n_hash_funcs
        self.n_tables = n_tables
        self.recall_k = max(top_k, recall_k)
        self.random_state = random_state
        self._memory_flat: np.ndarray | None = None
        self._proj: list[np.ndarray] = []
        self._tables: list[dict[int, np.ndarray]] = []

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self._memory_flat = flatten_windows(self._transform_histories(train_histories))
        dim = self._memory_flat.shape[1]
        self._proj = []
        self._tables = []
        for t in range(self.n_tables):
            rng = np.random.RandomState(self.random_state + t)
            proj = rng.randn(self.n_hash_funcs, dim).astype(np.float32)
            self._proj.append(proj)
            bits = (self._memory_flat @ proj.T) >= 0
            keys = self._pack(bits)
            table = defaultdict(list)
            for i, key in enumerate(keys):
                table[int(key)].append(i)
            self._tables.append({k: np.asarray(v, dtype=np.int64) for k, v in table.items()})

    def _pack(self, bits: np.ndarray) -> np.ndarray:
        powers = (1 << np.arange(bits.shape[1], dtype=np.uint64))[None, :]
        return (bits.astype(np.uint64) * powers).sum(axis=1)

    def retrieve(self, query_histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = flatten_windows(self._transform_histories(query_histories))
        b = q.shape[0]
        candidate_ids = np.zeros((b, self.recall_k), dtype=np.int64)
        candidate_scores = np.full((b, self.recall_k), 1e6, dtype=np.float32)
        for i in range(b):
            pools = []
            for proj, table in zip(self._proj, self._tables):
                bits = (q[i:i + 1] @ proj.T) >= 0
                key = int(self._pack(bits)[0])
                ids = table.get(key)
                if ids is not None:
                    pools.append(ids)
            if not pools:
                # fallback to global exact search
                d = l2_distance_matrix(q[i:i + 1], self._memory_flat)
                idx, vals = topk_from_distances(d, self.recall_k)
                candidate_ids[i] = idx[0]
                candidate_scores[i] = vals[0]
                continue

            ids = np.unique(np.concatenate(pools))
            d = l2_distance_matrix(q[i:i + 1], self._memory_flat[ids])[0]
            order = np.argsort(d)[: self.recall_k]
            chosen = ids[order]
            vals = d[order]
            fill = min(self.recall_k, len(chosen))
            candidate_ids[i, :fill] = chosen[:fill]
            candidate_scores[i, :fill] = vals[:fill]
            if fill < self.recall_k:
                candidate_ids[i, fill:] = chosen[0] if fill > 0 else 0
                candidate_scores[i, fill:] = vals[0] if fill > 0 else 1e6

        return candidate_ids, candidate_scores

    def rerank(self, query_histories: np.ndarray, candidate_ids: np.ndarray, candidate_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = flatten_windows(self._transform_histories(query_histories))
        return exact_l2_rerank(q, self._memory_flat, candidate_ids, self.top_k)
