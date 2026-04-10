from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from retrieval.aggregation import aggregate_futures
from retrieval.memory_bank import MemoryBank
from retrieval.rerank import hybrid_rerank


class BaseRetrieverForecaster(ABC):
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
        distance_mode: str = "all_channel_flat",
        channel_weights: Optional[list[float]] = None,
        target_idx: int = -1,
        rerank_mode: str = "none",
        rerank_alpha: float = 1.0,
        rerank_beta: float = 0.5,
        rerank_gamma: float = 0.3,
        rerank_delta: float = 0.2,
    ) -> None:
        if seq_len <= 0 or pred_len <= 0 or top_k <= 0:
            raise ValueError("seq_len, pred_len and top_k must be positive")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.top_k = int(top_k)

        self.normalization = normalization
        self.aggregation_mode = aggregation_mode
        self.aggregation_temperature = aggregation_temperature
        self.future_representation = future_representation
        self.restoration_mode = restoration_mode
        self.distance_mode = distance_mode
        self.channel_weights = channel_weights
        self.target_idx = target_idx

        self.rerank_mode = rerank_mode
        self.rerank_alpha = rerank_alpha
        self.rerank_beta = rerank_beta
        self.rerank_gamma = rerank_gamma
        self.rerank_delta = rerank_delta

        self.memory_bank = MemoryBank()
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted = False
        self.last_debug: dict = {}

    def build_memory_bank(self, train_histories: np.ndarray, train_futures: np.ndarray, train_phase: Optional[np.ndarray] = None) -> None:
        meta = {"phase": train_phase} if train_phase is not None else None
        self.memory_bank.build(train_histories, train_futures, meta=meta)

    def fit(
        self,
        train_histories: np.ndarray,
        train_futures: np.ndarray,
        train_phase: Optional[np.ndarray] = None,
    ) -> "BaseRetrieverForecaster":
        self.build_memory_bank(train_histories, train_futures, train_phase=train_phase)
        self._fit_normalizer(train_histories)
        self._fit_model(train_histories, train_futures)
        self._is_fitted = True
        return self

    def _query_stats(self, x: np.ndarray) -> dict:
        mean = x.mean(axis=1).astype(np.float32)
        std = np.clip(x.std(axis=1), 1e-6, None).astype(np.float32)
        last = x[:, -1, :].astype(np.float32)
        return {"mean": mean, "std": std, "last": last}

    def _select_future_rep(self, candidate_ids: np.ndarray) -> np.ndarray:
        if self.future_representation == "raw":
            return self.memory_bank.futures_raw[candidate_ids]
        if self.future_representation == "relative_norm":
            return self.memory_bank.futures_norm[candidate_ids]
        if self.future_representation == "delta":
            return self.memory_bank.futures_delta[candidate_ids]
        raise ValueError(f"unsupported future_representation: {self.future_representation}")

    def _restore_prediction(self, pred_repr: np.ndarray, q_stats: dict) -> np.ndarray:
        mode = self.restoration_mode
        if mode == "auto":
            mode = self.future_representation

        if mode in ("raw", "none"):
            return pred_repr
        if mode in ("relative_norm", "history_stat_norm"):
            return pred_repr * q_stats["std"][:, None, :] + q_stats["mean"][:, None, :]
        if mode == "delta":
            return pred_repr + q_stats["last"][:, None, :]
        raise ValueError(f"unsupported restoration mode: {mode}")

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

    def _channel_weights(self, c: int) -> np.ndarray:
        if self.channel_weights is not None:
            w = np.asarray(self.channel_weights, dtype=np.float32)
            if w.size != c:
                raise ValueError("channel_weights length mismatch")
            idx = self.target_idx if self.target_idx >= 0 else c - 1
            if w[idx] < np.max(w):
                # Ensure target channel is not accidentally under-weighted.
                w[idx] = np.max(w)
            return w
        w = np.ones(c, dtype=np.float32)
        idx = self.target_idx if self.target_idx >= 0 else c - 1
        idx = max(0, min(c - 1, idx))
        w[idx] = 3.0
        return w

    def vectorize_distance(self, x_transformed: np.ndarray) -> np.ndarray:
        # x_transformed: [N,L,C]
        n, l, c = x_transformed.shape
        if self.distance_mode == "target_only":
            idx = self.target_idx if self.target_idx >= 0 else c - 1
            return x_transformed[:, :, idx].reshape(n, -1).astype(np.float32)

        if self.distance_mode == "all_channel_flat":
            return x_transformed.reshape(n, -1).astype(np.float32)

        if self.distance_mode == "weighted_channel":
            w = np.sqrt(self._channel_weights(c))[None, None, :]
            return (x_transformed * w).reshape(n, -1).astype(np.float32)

        if self.distance_mode == "summary_augmented":
            shape = x_transformed.reshape(n, -1)
            mean = x_transformed.mean(axis=1)
            std = x_transformed.std(axis=1)
            last = x_transformed[:, -1, :]
            slope = (x_transformed[:, -1, :] - x_transformed[:, 0, :]) / max(1, l - 1)
            return np.concatenate([shape, mean, std, last, slope], axis=1).astype(np.float32)

        raise ValueError(f"unsupported distance_mode: {self.distance_mode}")

    def _target_candidate_distance(self, query_histories: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
        c = query_histories.shape[2]
        idx = self.target_idx if self.target_idx >= 0 else c - 1
        q = query_histories[:, :, idx]
        m = self.memory_bank.histories[:, :, idx]
        out = np.zeros(candidate_ids.shape, dtype=np.float32)
        for i in range(candidate_ids.shape[0]):
            cand = m[candidate_ids[i]]
            diff = cand - q[i:i + 1]
            out[i] = np.sqrt(np.mean(diff * diff, axis=1))
        return out

    def forecast(self, query_histories: np.ndarray, query_phase: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("model is not fitted")

        q_stats = self._query_stats(query_histories)
        candidate_ids, scores = self.retrieve(query_histories, query_phase=query_phase)
        pre_ids = candidate_ids.copy()
        pre_scores = scores.copy()
        pre_target_dist = self._target_candidate_distance(query_histories, pre_ids)

        if self.rerank_mode == "hybrid":
            reranked_ids, reranked_scores, decomp = hybrid_rerank(
                candidate_ids=candidate_ids,
                shape_scores=scores,
                query_mean=q_stats["mean"],
                query_std=q_stats["std"],
                query_last=q_stats["last"],
                memory_mean=self.memory_bank.hist_mean,
                memory_std=self.memory_bank.hist_std,
                memory_last=self.memory_bank.hist_last,
                query_phase=query_phase,
                memory_phase=self.memory_bank.phase,
                top_k=self.top_k,
                alpha=self.rerank_alpha,
                beta=self.rerank_beta,
                gamma=self.rerank_gamma,
                delta=self.rerank_delta,
            )
        else:
            reranked_ids, reranked_scores = self.rerank(query_histories, candidate_ids, scores, query_phase=query_phase)
            decomp = None
        post_target_dist = self._target_candidate_distance(query_histories, reranked_ids)

        candidate_future_repr = self._select_future_rep(reranked_ids)
        pred_repr, weights, agg_stats = aggregate_futures(
            candidate_future_repr,
            reranked_scores,
            mode=self.aggregation_mode,
            temperature=self.aggregation_temperature,
        )
        pred = self._restore_prediction(pred_repr, q_stats)

        self.last_debug = {
            "pre_candidate_ids": pre_ids,
            "pre_candidate_scores": pre_scores,
            "candidate_ids": reranked_ids,
            "candidate_scores": reranked_scores,
            "pre_target_dist": pre_target_dist,
            "post_target_dist": post_target_dist,
            "candidate_changed_ratio": float(np.mean(np.any(pre_ids[:, : self.top_k] != reranked_ids, axis=1))),
            "weights": weights,
            "agg_stats": agg_stats,
            "query_mean": q_stats["mean"],
            "query_last": q_stats["last"],
            "candidate_hist_mean": self.memory_bank.hist_mean[reranked_ids],
            "query_phase": query_phase,
            "candidate_phase": self.memory_bank.phase[reranked_ids] if self.memory_bank.phase is not None else None,
            "rerank_decomp": decomp,
        }

        return pred.astype(np.float32)

    def predict(self, query_histories: np.ndarray, query_phase: Optional[np.ndarray] = None) -> np.ndarray:
        return self.forecast(query_histories, query_phase=query_phase)

    @abstractmethod
    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        pass

    @abstractmethod
    def retrieve(self, query_histories: np.ndarray, query_phase: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        pass

    def rerank(
        self,
        query_histories: np.ndarray,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
        query_phase: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = min(self.top_k, candidate_ids.shape[1])
        return candidate_ids[:, :k], candidate_scores[:, :k]
