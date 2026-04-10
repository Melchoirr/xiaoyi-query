from __future__ import annotations

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.distance import dtw_distance


class DTWSearch(BaseRetrieverForecaster):
    """Elastic-shape retrieval forecaster using constrained DTW."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
        dtw_radius: int = 5,
        normalization: str = "window_zscore",
        aggregation_mode: str = "inverse_distance",
        aggregation_temperature: float = 1.0,
        future_representation: str = "relative_norm",
        restoration_mode: str = "auto",
        distance_mode: str = "weighted_channel",
        channel_weights: list[float] | None = None,
        target_idx: int = -1,
        rerank_mode: str = "none",
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
        self.dtw_radius = dtw_radius
        self._memory_hist: np.ndarray | None = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self._memory_hist = self._transform_histories(train_histories)

    def _apply_distance_mode(self, x: np.ndarray) -> np.ndarray:
        c = x.shape[2]
        if self.distance_mode == "target_only":
            idx = self.target_idx if self.target_idx >= 0 else c - 1
            return x[:, :, idx:idx + 1]
        if self.distance_mode == "all_channel_flat":
            return x
        if self.distance_mode == "weighted_channel":
            w = np.sqrt(self._channel_weights(c))[None, None, :]
            return x * w
        if self.distance_mode == "summary_augmented":
            return x
        raise ValueError(f"unsupported distance_mode: {self.distance_mode}")

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None):
        q = self._apply_distance_mode(self._transform_histories(query_histories))
        m = self._apply_distance_mode(self._memory_hist)
        n = q.shape[0]
        mm = m.shape[0]
        scores = np.zeros((n, mm), dtype=np.float32)
        for i in range(n):
            for j in range(mm):
                scores[i, j] = dtw_distance(q[i], m[j], radius=self.dtw_radius)

        k = min(self.top_k, mm)
        idx = np.argpartition(scores, kth=k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        return idx.astype(np.int64), vals.astype(np.float32)
