from __future__ import annotations

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import flatten_windows, l2_distance_matrix


class PatternSearch(BaseRetrieverForecaster):
    """Exact distance-based retrieval forecaster.

    Normalization policy: window-level z-score by default.
    """

    def __init__(self, seq_len: int, pred_len: int, top_k: int = 5, normalization: str = "window_zscore") -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            top_k=top_k,
            normalization=normalization,
            aggregation="inverse_distance",
        )
        self._memory_flat: np.ndarray | None = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        norm_hist = self._transform_histories(train_histories)
        self._memory_flat = flatten_windows(norm_hist)

    def retrieve(self, query_histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query_norm = self._transform_histories(query_histories)
        qf = flatten_windows(query_norm)
        d = l2_distance_matrix(qf, self._memory_flat)
        return topk_from_distances(d, self.top_k)
