from __future__ import annotations

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import l2_distance_matrix


class MatrixProfileSearch(BaseRetrieverForecaster):
    """z-normalized subsequence nearest-neighbor forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
        normalization: str = "window_zscore",
        aggregation_mode: str = "inverse_distance",
        future_representation: str = "relative_norm",
        restoration_mode: str = "auto",
    ) -> None:
        super().__init__(
            seq_len,
            pred_len,
            top_k,
            normalization=normalization,
            aggregation_mode=aggregation_mode,
            future_representation=future_representation,
            restoration_mode=restoration_mode,
            distance_mode="all_channel_flat",
        )
        self._memory_flat: np.ndarray | None = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self._memory_flat = self.vectorize_distance(self._transform_histories(train_histories))

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None):
        q = self.vectorize_distance(self._transform_histories(query_histories))
        d = l2_distance_matrix(q, self._memory_flat)
        return topk_from_distances(d, self.top_k)
