from __future__ import annotations

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import flatten_windows, l2_distance_matrix


class MatrixProfileSearch(BaseRetrieverForecaster):
    """z-normalized subsequence nearest-neighbor forecaster.

    Note: this class implements fixed-length z-normalized subsequence search,
    not full matrix-profile motif discovery.
    """

    def __init__(self, seq_len: int, pred_len: int, top_k: int = 5, normalization: str = "window_zscore") -> None:
        super().__init__(seq_len, pred_len, top_k, normalization=normalization, aggregation="inverse_distance")
        self._memory_flat: np.ndarray | None = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self._memory_flat = flatten_windows(self._transform_histories(train_histories))

    def retrieve(self, query_histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = flatten_windows(self._transform_histories(query_histories))
        d = l2_distance_matrix(q, self._memory_flat)
        return topk_from_distances(d, self.top_k)
