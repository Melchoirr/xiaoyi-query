from __future__ import annotations

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import l2_distance_matrix


class PatternSearch(BaseRetrieverForecaster):
    """Exact distance retrieval forecaster with relative-future restoration support."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
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
        self._memory_vec: np.ndarray | None = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        hist = self._transform_histories(train_histories)
        self._memory_vec = self.vectorize_distance(hist)

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = self._transform_histories(query_histories)
        qv = self.vectorize_distance(q)
        d = l2_distance_matrix(qv, self._memory_vec)
        return topk_from_distances(d, self.top_k)
