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
    ) -> None:
        super().__init__(seq_len, pred_len, top_k, normalization=normalization, aggregation="inverse_distance")
        self.dtw_radius = dtw_radius
        self._memory_hist: np.ndarray | None = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self._memory_hist = self._transform_histories(train_histories)

    def retrieve(self, query_histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self._transform_histories(query_histories)
        n = q.shape[0]
        m = self._memory_hist.shape[0]
        scores = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                scores[i, j] = dtw_distance(q[i], self._memory_hist[j], radius=self.dtw_radius)

        idx = np.argpartition(scores, kth=min(self.top_k, m) - 1, axis=1)[:, : self.top_k]
        vals = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        return idx.astype(np.int64), vals.astype(np.float32)
