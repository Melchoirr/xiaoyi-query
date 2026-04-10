from __future__ import annotations

import numpy as np

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import l2_distance_matrix


class RepeatLastValue(BaseRetrieverForecaster):
    def __init__(self, seq_len: int, pred_len: int, top_k: int = 1):
        super().__init__(seq_len, pred_len, top_k=1, normalization="none", future_representation="raw")

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        return

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None):
        b = query_histories.shape[0]
        return np.zeros((b, 1), dtype=np.int64), np.zeros((b, 1), dtype=np.float32)

    def forecast(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None) -> np.ndarray:
        last = query_histories[:, -1:, :]
        return np.repeat(last, self.pred_len, axis=1).astype(np.float32)


class HistoricalMean(BaseRetrieverForecaster):
    def __init__(self, seq_len: int, pred_len: int, top_k: int = 1):
        super().__init__(seq_len, pred_len, top_k=1, normalization="none", future_representation="raw")

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self.global_mean_future = train_futures.mean(axis=0).astype(np.float32)

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None):
        b = query_histories.shape[0]
        return np.zeros((b, 1), dtype=np.int64), np.zeros((b, 1), dtype=np.float32)

    def forecast(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None) -> np.ndarray:
        b = query_histories.shape[0]
        return np.repeat(self.global_mean_future[None, :, :], b, axis=0).astype(np.float32)


class Top1NearestFuture(BaseRetrieverForecaster):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__(seq_len, pred_len, top_k=1, normalization="window_zscore", future_representation="raw", restoration_mode="raw")
        self.mem = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self.mem = self.vectorize_distance(self._transform_histories(train_histories))

    def retrieve(self, query_histories: np.ndarray, query_phase: np.ndarray | None = None):
        q = self.vectorize_distance(self._transform_histories(query_histories))
        d = l2_distance_matrix(q, self.mem)
        return topk_from_distances(d, 1)
