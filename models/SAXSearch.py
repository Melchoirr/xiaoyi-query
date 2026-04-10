from __future__ import annotations

from collections import defaultdict

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.distance import l2_distance_matrix
from retrieval.rerank import exact_l2_rerank


class SAXSearch(BaseRetrieverForecaster):
    """SAX coarse recall -> exact rerank -> relative-future aggregation."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
        word_size: int = 8,
        alphabet_size: int = 8,
        recall_k: int = 64,
        normalization: str = "window_zscore",
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
        if word_size <= 0 or alphabet_size <= 1:
            raise ValueError("invalid SAX hyperparameters")
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.recall_k = max(recall_k, top_k)
        self._memory_vec: np.ndarray | None = None
        self._bucket_to_ids: dict[tuple[int, ...], np.ndarray] = {}
        self._bucket_centroids: dict[tuple[int, ...], np.ndarray] = {}
        self._breakpoints = np.linspace(-2.0, 2.0, alphabet_size - 1, dtype=np.float32)

    def _paa(self, x: np.ndarray) -> np.ndarray:
        b, l = x.shape
        bins = np.array_split(np.arange(l), self.word_size)
        return np.stack([x[:, idx].mean(axis=1) for idx in bins], axis=1).astype(np.float32)

    def _encode(self, x_vec: np.ndarray) -> np.ndarray:
        paa = self._paa(x_vec)
        return np.digitize(paa, self._breakpoints).astype(np.int64)

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self._memory_vec = self.vectorize_distance(self._transform_histories(train_histories))
        tokens = self._encode(self._memory_vec)
        buckets = defaultdict(list)
        for i, token in enumerate(tokens):
            buckets[tuple(token.tolist())].append(i)
        self._bucket_to_ids = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
        self._bucket_centroids = {k: self._memory_vec[v].mean(axis=0).astype(np.float32) for k, v in self._bucket_to_ids.items()}

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None):
        qv = self.vectorize_distance(self._transform_histories(query_histories))
        qt = self._encode(qv)

        cand_ids = np.zeros((qv.shape[0], self.recall_k), dtype=np.int64)
        cand_scores = np.full((qv.shape[0], self.recall_k), 1e6, dtype=np.float32)
        centroid_keys = list(self._bucket_centroids.keys())
        centroid_vals = np.stack([self._bucket_centroids[k] for k in centroid_keys], axis=0)

        for i in range(qv.shape[0]):
            key = tuple(qt[i].tolist())
            ids = self._bucket_to_ids.get(key)
            if ids is None:
                d_cent = l2_distance_matrix(qv[i:i + 1], centroid_vals)[0]
                nearest_bucket = centroid_keys[int(np.argmin(d_cent))]
                ids = self._bucket_to_ids[nearest_bucket]
            d = l2_distance_matrix(qv[i:i + 1], self._memory_vec[ids])[0]
            order = np.argsort(d)[: self.recall_k]
            chosen = ids[order]
            vals = d[order]
            fill = min(len(chosen), self.recall_k)
            cand_ids[i, :fill] = chosen[:fill]
            cand_scores[i, :fill] = vals[:fill]
            if fill < self.recall_k:
                cand_ids[i, fill:] = chosen[0] if fill > 0 else 0
                cand_scores[i, fill:] = vals[0] if fill > 0 else 1e6
        return cand_ids, cand_scores

    def rerank(self, query_histories: np.ndarray, candidate_ids: np.ndarray, candidate_scores: np.ndarray, query_phase: np.ndarray | None = None):
        if self.rerank_mode == "none":
            return candidate_ids[:, : self.top_k], candidate_scores[:, : self.top_k]
        qv = self.vectorize_distance(self._transform_histories(query_histories))
        return exact_l2_rerank(qv, self._memory_vec, candidate_ids, self.top_k)
