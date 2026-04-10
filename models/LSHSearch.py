from __future__ import annotations

from collections import defaultdict

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import l2_distance_matrix
from retrieval.rerank import exact_l2_rerank


class LSHSearch(BaseRetrieverForecaster):
    """Approximate recall -> exact rerank -> relative-future aggregation."""

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
        aggregation_mode: str = "inverse_distance",
        aggregation_temperature: float = 1.0,
        future_representation: str = "relative_norm",
        restoration_mode: str = "auto",
        distance_mode: str = "weighted_channel",
        channel_weights: list[float] | None = None,
        target_idx: int = -1,
        rerank_mode: str = "exact",
        rerank_alpha: float = 1.0,
        rerank_beta: float = 0.5,
        rerank_gamma: float = 0.3,
        rerank_delta: float = 0.2,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            top_k=top_k,
            normalization=normalization,
            aggregation_mode=aggregation_mode,
            aggregation_temperature=aggregation_temperature,
            future_representation=future_representation,
            restoration_mode=restoration_mode,
            distance_mode=distance_mode,
            channel_weights=channel_weights,
            target_idx=target_idx,
            rerank_mode=rerank_mode,
            rerank_alpha=rerank_alpha,
            rerank_beta=rerank_beta,
            rerank_gamma=rerank_gamma,
            rerank_delta=rerank_delta,
        )
        self.n_hash_funcs = n_hash_funcs
        self.n_tables = n_tables
        self.recall_k = max(top_k, recall_k)
        self.random_state = random_state
        self._memory_vec: np.ndarray | None = None
        self._proj: list[np.ndarray] = []
        self._tables: list[dict[int, np.ndarray]] = []

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self._memory_vec = self.vectorize_distance(self._transform_histories(train_histories))
        dim = self._memory_vec.shape[1]
        self._proj = []
        self._tables = []
        for t in range(self.n_tables):
            rng = np.random.RandomState(self.random_state + t)
            proj = rng.randn(self.n_hash_funcs, dim).astype(np.float32)
            self._proj.append(proj)
            bits = (self._memory_vec @ proj.T) >= 0
            keys = self._pack(bits)
            table = defaultdict(list)
            for i, key in enumerate(keys):
                table[int(key)].append(i)
            self._tables.append({k: np.asarray(v, dtype=np.int64) for k, v in table.items()})

    def _pack(self, bits: np.ndarray) -> np.ndarray:
        powers = (1 << np.arange(bits.shape[1], dtype=np.uint64))[None, :]
        return (bits.astype(np.uint64) * powers).sum(axis=1)

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        qv = self.vectorize_distance(self._transform_histories(query_histories))
        b = qv.shape[0]
        candidate_ids = np.zeros((b, self.recall_k), dtype=np.int64)
        candidate_scores = np.full((b, self.recall_k), 1e6, dtype=np.float32)
        for i in range(b):
            pools = []
            for proj, table in zip(self._proj, self._tables):
                bits = (qv[i:i + 1] @ proj.T) >= 0
                key = int(self._pack(bits)[0])
                ids = table.get(key)
                if ids is not None:
                    pools.append(ids)
            if not pools:
                d = l2_distance_matrix(qv[i:i + 1], self._memory_vec)
                idx, vals = topk_from_distances(d, self.recall_k)
                candidate_ids[i] = idx[0]
                candidate_scores[i] = vals[0]
                continue

            ids = np.unique(np.concatenate(pools))
            d = l2_distance_matrix(qv[i:i + 1], self._memory_vec[ids])[0]
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

    def rerank(self, query_histories: np.ndarray, candidate_ids: np.ndarray, candidate_scores: np.ndarray, query_phase: np.ndarray | None = None):
        if self.rerank_mode == "none":
            return candidate_ids[:, : self.top_k], candidate_scores[:, : self.top_k]
        qv = self.vectorize_distance(self._transform_histories(query_histories))
        return exact_l2_rerank(qv, self._memory_vec, candidate_ids, self.top_k)
