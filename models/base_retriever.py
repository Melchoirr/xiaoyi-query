from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from retrieval.aggregation import aggregate_futures
from retrieval.memory_bank import MemoryBank


class BaseRetrieverForecaster(ABC):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
        normalization: str = "window_zscore",
        aggregation: str = "inverse_distance",
    ) -> None:
        if seq_len <= 0 or pred_len <= 0 or top_k <= 0:
            raise ValueError("seq_len, pred_len and top_k must be positive")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.top_k = int(top_k)
        self.normalization = normalization
        self.aggregation = aggregation
        self.memory_bank = MemoryBank()
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted = False

    def build_memory_bank(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        self.memory_bank.build(train_histories, train_futures)

    def fit(self, train_histories: np.ndarray, train_futures: np.ndarray) -> "BaseRetrieverForecaster":
        self.build_memory_bank(train_histories, train_futures)
        self._fit_normalizer(train_histories)
        self._fit_model(train_histories, train_futures)
        self._is_fitted = True
        return self

    def forecast(self, query_histories: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("model is not fitted")
        candidate_ids, scores = self.retrieve(query_histories)
        reranked_ids, reranked_scores = self.rerank(query_histories, candidate_ids, scores)
        candidate_futures = self.memory_bank.futures[reranked_ids]
        return self.aggregate_future(candidate_futures, reranked_scores)

    def predict(self, query_histories: np.ndarray) -> np.ndarray:
        return self.forecast(query_histories)

    def _fit_normalizer(self, train_histories: np.ndarray) -> None:
        if self.normalization == "standard":
            flat = train_histories.reshape(-1, train_histories.shape[-1])
            self._scaler = StandardScaler()
            self._scaler.fit(flat)

    def _transform_histories(self, x: np.ndarray) -> np.ndarray:
        if self.normalization == "none":
            return x.astype(np.float32, copy=False)
        if self.normalization == "window_zscore":
            mean = x.mean(axis=1, keepdims=True)
            std = np.clip(x.std(axis=1, keepdims=True), 1e-8, None)
            return ((x - mean) / std).astype(np.float32)
        if self.normalization == "standard":
            if self._scaler is None:
                raise RuntimeError("standard scaler has not been fitted")
            shp = x.shape
            flat = x.reshape(-1, shp[-1])
            out = self._scaler.transform(flat).reshape(shp)
            return out.astype(np.float32)
        raise ValueError(f"unsupported normalization: {self.normalization}")

    @abstractmethod
    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        pass

    @abstractmethod
    def retrieve(self, query_histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    def rerank(
        self,
        query_histories: np.ndarray,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return candidate_ids, candidate_scores

    def aggregate_future(self, candidate_futures: np.ndarray, scores: np.ndarray) -> np.ndarray:
        return aggregate_futures(candidate_futures, scores, mode=self.aggregation).astype(np.float32)
