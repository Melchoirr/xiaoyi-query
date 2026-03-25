"""
SAXSearch: symbolic aggregate approximation baseline for time-series retrieval.

This version keeps SAX's compressed bucket-level prediction design, while
replacing the previous full-vocabulary edit-distance scan with nearest-bucket
lookup in low-dimensional PAA space via sklearn.neighbors.NearestNeighbors.
"""

import gc
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class SAXSearch:
    DTYPE = np.float32

    def __init__(
        self,
        word_size: int = 8,
        alphabet_size: int = 8,
        epsilon_threshold: float = 1.0,
        fallback_strategy: str = 'global_mean',
        random_state: Optional[int] = 42,
        seq_len: int = 0,
        pred_len: int = 0,
        n_features: int = 1,
        device: Union[str, torch.device] = 'cpu',
        bucket_top_k: int = 8,
        weighted: bool = True,
        **kwargs
    ):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.epsilon_threshold = epsilon_threshold
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.bucket_top_k = max(1, int(bucket_top_k))
        self.weighted = weighted

        self.breakpoints = self._compute_breakpoints()
        self._base_powers = (self.alphabet_size ** np.arange(self.word_size, dtype=np.int64)).astype(np.int64)

        self.sax_dict: Dict[int, Tuple[np.ndarray, int]] = {}
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted = False
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len

        self.bucket_codes: Optional[np.ndarray] = None
        self.bucket_mean_paa: Optional[np.ndarray] = None
        self.bucket_mean_y: Optional[np.ndarray] = None
        self.bucket_counts: Optional[np.ndarray] = None
        self._nn_index: Optional[NearestNeighbors] = None

    def _compute_breakpoints(self) -> np.ndarray:
        breakpoints = norm.ppf(np.linspace(
            1.0 / self.alphabet_size,
            1.0 - 1.0 / self.alphabet_size,
            self.alphabet_size - 1
        ))
        return breakpoints.astype(self.DTYPE)

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.device.type == 'cuda':
            t = torch.from_numpy(X).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                X_mean = t.mean(dim=1, keepdim=True)
                X_std = t.std(dim=1, keepdim=True).clamp(min=1e-8)
                X_norm = ((t - X_mean) / X_std).cpu().numpy().astype(self.DTYPE)
            return X_norm, np.zeros((X.shape[0], 1), dtype=self.DTYPE), np.ones((X.shape[0], 1), dtype=self.DTYPE)

        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_std = np.clip(X_std, 1e-8, None)
        X_norm = (X - X_mean) / X_std
        return X_norm.astype(self.DTYPE), X_mean.astype(self.DTYPE), X_std.astype(self.DTYPE)

    def _paa_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len = X.shape
        seg_len = seq_len / self.word_size
        paa = np.zeros((n_samples, self.word_size), dtype=self.DTYPE)
        for i in range(self.word_size):
            start = int(i * seg_len)
            end = int((i + 1) * seg_len)
            paa[:, i] = np.mean(X[:, start:end], axis=1)
        return paa

    def _quantize_paa(self, paa: np.ndarray) -> np.ndarray:
        bp = self.breakpoints.astype(np.float64)
        if bp.size == 0:
            idx = np.zeros(paa.shape, dtype=np.int64)
        else:
            idx = (paa[..., None] > bp.reshape(1, 1, -1)).sum(axis=-1).astype(np.int64)
        return np.clip(idx, 0, self.alphabet_size - 1)

    def _pack_symbol_codes(self, symbol_idx: np.ndarray) -> np.ndarray:
        return (symbol_idx.astype(np.int64) * self._base_powers.reshape(1, -1)).sum(axis=1, dtype=np.int64)

    def _predict_from_bucket_neighbors(self, distances: np.ndarray, neighbor_idx: np.ndarray) -> Optional[np.ndarray]:
        valid = np.isfinite(distances)
        if self.epsilon_threshold > 0:
            max_distance = float(self.epsilon_threshold) * float(np.sqrt(self.word_size))
            valid = valid & (distances <= max_distance)
        if not np.any(valid):
            return None

        dist = distances[valid].astype(self.DTYPE)
        neigh = neighbor_idx[valid]
        selected_y = self.bucket_mean_y[neigh]

        if self.weighted:
            weights = 1.0 / np.maximum(dist, 1e-8)
            weights = weights / np.sum(weights)
            return np.tensordot(weights.astype(self.DTYPE), selected_y, axes=(0, 0)).astype(self.DTYPE)

        return selected_y.mean(axis=0).astype(self.DTYPE)

    # ─────────────────────────────────────────────────────────
    #  fit()
    # ─────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        # ── 严格捕获真实维度，禁止 n_features=1 残留 ──────────────────
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        self.n_features = Y_train.shape[-1] if Y_train.ndim == 3 else 1

        n_samples = X_train.shape[0]

        if X_train.ndim == 3:
            X_flat = X_train.reshape(n_samples, -1).astype(self.DTYPE)
            Y_original = Y_train.astype(self.DTYPE)
        else:
            X_flat = X_train.astype(self.DTYPE)
            Y_original = Y_train.astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)
        paa = self._paa_transform(X_norm)
        symbol_idx = self._quantize_paa(paa)
        packed_codes = self._pack_symbol_codes(symbol_idx)

        sum_y_cache: Dict[int, np.ndarray] = {}
        sum_paa_cache: Dict[int, np.ndarray] = {}
        count_cache: Dict[int, int] = {}

        for i in range(n_samples):
            code = int(packed_codes[i])
            if code not in sum_y_cache:
                sum_y_cache[code] = np.zeros((self.pred_len, self.n_features), dtype=self.DTYPE)
                sum_paa_cache[code] = np.zeros(self.word_size, dtype=self.DTYPE)
                count_cache[code] = 0
            sum_y_cache[code] += Y_original[i]
            sum_paa_cache[code] += paa[i]
            count_cache[code] += 1

        bucket_codes: List[int] = []
        bucket_mean_paa: List[np.ndarray] = []
        bucket_mean_y: List[np.ndarray] = []
        bucket_counts: List[int] = []
        self.sax_dict = {}

        for code in sum_y_cache:
            cnt = count_cache[code]
            mean_y = (sum_y_cache[code] / cnt).astype(self.DTYPE)
            mean_paa = (sum_paa_cache[code] / cnt).astype(self.DTYPE)
            self.sax_dict[code] = (mean_y, cnt)
            bucket_codes.append(code)
            bucket_mean_paa.append(mean_paa)
            bucket_mean_y.append(mean_y)
            bucket_counts.append(cnt)

        self.bucket_codes = np.asarray(bucket_codes, dtype=np.int64)
        self.bucket_mean_paa = np.asarray(bucket_mean_paa, dtype=self.DTYPE)
        self.bucket_mean_y = np.asarray(bucket_mean_y, dtype=self.DTYPE)
        self.bucket_counts = np.asarray(bucket_counts, dtype=np.int32)

        n_neighbors = min(self.bucket_top_k, len(self.bucket_codes))
        self._nn_index = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='auto',
            metric='euclidean',
        )
        self._nn_index.fit(self.bucket_mean_paa)

        self.global_Y_mean = np.mean(Y_original, axis=0).astype(self.DTYPE)
        del Y_original, X_flat, X_norm, paa, symbol_idx, packed_codes
        del sum_y_cache, sum_paa_cache, count_cache
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.is_fitted = True

        logger.info(
            f"[SAXSearch] fitted: {len(self.sax_dict)} unique buckets, "
            f"nn_top_k={n_neighbors}, device={self.device}"
        )
        return self

    # ─────────────────────────────────────────────────────────
    #  predict()
    # ─────────────────────────────────────────────────────────

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        if not self.is_fitted or self._nn_index is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]

        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            X_flat = X_test.astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)
        paa = self._paa_transform(X_norm)
        symbol_idx = self._quantize_paa(paa)
        packed_codes = self._pack_symbol_codes(symbol_idx)

        Y_pred = np.zeros((n_test, self.pred_len, self.n_features), dtype=self.DTYPE)

        matched = np.zeros(n_test, dtype=bool)

        exact_hit = 0
        for i in range(n_test):
            code = int(packed_codes[i])
            bucket = self.sax_dict.get(code)
            if bucket is not None:
                Y_pred[i] = bucket[0]
                matched[i] = True
                exact_hit += 1

        fuzzy_hit = 0
        fallback_count = 0
        todo = np.nonzero(~matched)[0]

        if len(todo) > 0 and len(self.bucket_codes) > 0 and self.epsilon_threshold > 0:
            query_paa = paa[todo]
            n_neighbors = min(self.bucket_top_k, len(self.bucket_codes))
            distances, neighbor_idx = self._nn_index.kneighbors(
                query_paa, n_neighbors=n_neighbors, return_distance=True
            )

            for local_row, sample_idx in enumerate(todo):
                pred = self._predict_from_bucket_neighbors(distances[local_row], neighbor_idx[local_row])
                if pred is not None:
                    Y_pred[sample_idx] = pred
                    matched[sample_idx] = True
                    fuzzy_hit += 1
                else:
                    fallback_count += 1
                    if self.fallback_strategy == 'global_mean':
                        Y_pred[sample_idx] = self.global_Y_mean
                    else:
                        last_vals = X_flat[sample_idx].reshape(self.seq_len, self.n_features)
                        Y_pred[sample_idx] = np.full(
                            (self.pred_len, self.n_features),
                            float(np.mean(last_vals[-1])),
                            dtype=self.DTYPE,
                        )
        elif len(todo) > 0:
            for sample_idx in todo:
                fallback_count += 1
                if self.fallback_strategy == 'global_mean':
                    Y_pred[sample_idx] = self.global_Y_mean
                else:
                    last_vals = X_flat[sample_idx].reshape(self.seq_len, self.n_features)
                    Y_pred[sample_idx] = np.full(
                        (self.pred_len, self.n_features),
                        float(np.mean(last_vals[-1])),
                        dtype=self.DTYPE,
                    )

        if fallback_count > 0:
            logger.warning(f"[SAXSearch] fallback {fallback_count}/{n_test}")

        logger.info(
            f"[SAXSearch] predict done {Y_pred.shape}, exact_hit={exact_hit}, "
            f"fuzzy_hit={fuzzy_hit}, device={self.device}"
        )

        if self.n_features == 1:
            Y_pred = Y_pred.squeeze(-1)

        del X_flat, X_norm, paa, symbol_idx, packed_codes
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return Y_pred

    # ─────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            'word_size': self.word_size,
            'alphabet_size': self.alphabet_size,
            'epsilon_threshold': self.epsilon_threshold,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'bucket_top_k': self.bucket_top_k,
            'weighted': self.weighted,
        }

    def get_sax_stats(self) -> dict:
        if not self.is_fitted or self.bucket_counts is None:
            return {}
        total_samples = int(np.sum(self.bucket_counts))
        return {
            'n_unique_strings': len(self.sax_dict),
            'total_samples': total_samples,
            'min_bucket': int(np.min(self.bucket_counts)) if self.bucket_counts.size else 0,
            'max_bucket': int(np.max(self.bucket_counts)) if self.bucket_counts.size else 0,
            'avg_bucket': float(np.mean(self.bucket_counts)) if self.bucket_counts.size else 0,
        }

    def __repr__(self):
        return (
            f"SAXSearch(word_size={self.word_size}, alphabet_size={self.alphabet_size}, "
            f"epsilon_threshold={self.epsilon_threshold}, bucket_top_k={self.bucket_top_k}, "
            f"weighted={self.weighted})"
        )
