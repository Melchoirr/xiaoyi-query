"""
LSHSearch: 局部敏感哈希时序预测基线
predict 阶段对全体测试样本批量 torch 投影，再查表聚合（消除逐样本 matmul）
"""

import gc
import logging
from typing import Optional, Tuple, Dict, Union, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LSHSearch:
    DTYPE = np.float32

    def __init__(
        self,
        n_hash_funcs: int = 16,
        n_tables: int = 4,
        hamming_radius: int = 2,
        fallback_strategy: str = 'global_mean',
        random_state: Optional[int] = 42,
        seq_len: int = 0,
        pred_len: int = 0,
        n_features: int = 1,
        device: Union[str, torch.device] = 'cpu',
        **kwargs
    ):
        self.n_hash_funcs = n_hash_funcs
        self.n_tables = n_tables
        self.hamming_radius = hamming_radius
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.hash_tables: List[Dict[str, Tuple[np.ndarray, int]]] = []
        self.projection_matrices: List[np.ndarray] = []
        self._proj_t_list: List[torch.Tensor] = []
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted = False
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X.ndim == 3:
            X_mean = np.mean(X, axis=1, keepdims=True)
            X_std = np.std(X, axis=1, keepdims=True)
        else:
            X_mean = np.mean(X, axis=1, keepdims=True)
            X_std = np.std(X, axis=1, keepdims=True)
        X_std = np.clip(X_std, 1e-8, None)
        X_norm = (X - X_mean) / X_std
        return X_norm, X_mean, X_std

    def _generate_projection_matrix(self, dim: int) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        projection_matrix = rng.randn(self.n_hash_funcs, dim).astype(self.DTYPE)
        norms = np.linalg.norm(projection_matrix, axis=1, keepdims=True)
        projection_matrix = projection_matrix / (norms + 1e-10)
        return projection_matrix

    def _hash_code_to_key(self, hash_code: np.ndarray) -> str:
        return ''.join(map(str, hash_code.tolist()))

    def _hamming_distance(self, code1: np.ndarray, code2: np.ndarray) -> int:
        return int(np.sum(code1 != code2))

    def _find_bucket_mean(
        self,
        query_code: np.ndarray,
        bucket_key: str,
        table: Dict[str, Tuple[np.ndarray, int]]
    ) -> Tuple[Optional[np.ndarray], int]:
        if bucket_key in table:
            return table[bucket_key]
        if self.hamming_radius > 0:
            best_key = None
            best_dist = float('inf')
            for stored_key, val in table.items():
                if stored_key == bucket_key:
                    continue
                stored_code = np.array([int(c) for c in stored_key], dtype=np.uint8)
                dist = self._hamming_distance(query_code, stored_code)
                if dist < best_dist and dist <= self.hamming_radius:
                    best_dist = dist
                    best_key = stored_key
            if best_key is not None:
                return table[best_key]
        return (None, 0)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        n_samples = X_train.shape[0]

        if X_train.ndim == 3:
            self.seq_len = X_train.shape[1]
            self.n_features = X_train.shape[2]
            X_flat = X_train.reshape(n_samples, -1).astype(self.DTYPE)
        else:
            self.seq_len = X_train.shape[1]
            self.n_features = 1
            X_flat = X_train.reshape(n_samples, -1).astype(self.DTYPE)

        if Y_train.ndim == 3:
            self.pred_len = Y_train.shape[1]
            Y_original = Y_train.astype(self.DTYPE)
        else:
            self.pred_len = Y_train.shape[1]
            self.n_features = 1
            Y_original = Y_train.astype(self.DTYPE)

        self.hash_tables = []
        self.projection_matrices = []
        self._proj_t_list = []

        X_norm, _, _ = self._instance_normalize(X_flat)

        for table_idx in range(self.n_tables):
            original_seed = self.random_state
            self.random_state = original_seed + table_idx * 1000

            projection_matrix = self._generate_projection_matrix(X_flat.shape[1])
            self.projection_matrices.append(projection_matrix)
            self._proj_t_list.append(
                torch.from_numpy(projection_matrix).to(self.device, dtype=torch.float32)
            )

            with torch.no_grad():
                Xn_t = torch.from_numpy(X_norm).to(self.device, dtype=torch.float32)
                proj = Xn_t @ self._proj_t_list[-1].T
                hash_codes = (proj >= 0).cpu().numpy().astype(np.uint8)

            sum_cache: Dict[str, np.ndarray] = {}
            count_cache: Dict[str, int] = {}
            hash_table: Dict[str, Tuple[np.ndarray, int]] = {}

            for i in range(n_samples):
                key = self._hash_code_to_key(hash_codes[i])
                if key not in sum_cache:
                    sum_cache[key] = np.zeros(
                        (self.pred_len, self.n_features), dtype=self.DTYPE)
                    count_cache[key] = 0
                sum_cache[key] += Y_original[i]
                count_cache[key] += 1

            for key in sum_cache:
                cnt = count_cache[key]
                mean_Y = (sum_cache[key] / cnt).astype(self.DTYPE)
                hash_table[key] = (mean_Y, cnt)

            self.hash_tables.append(hash_table)
            self.random_state = original_seed

        self.global_Y_mean = np.mean(Y_original, axis=0).astype(self.DTYPE)
        del Y_original, X_flat, X_norm
        gc.collect()
        self.is_fitted = True

        n_buckets = sum(len(t) for t in self.hash_tables)
        logger.info(
            f"[LSHSearch] fitted device={self.device}, tables={self.n_tables}, buckets={n_buckets}"
        )
        return self

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]
        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)
        Y_pred = np.zeros((n_test, self.pred_len, self.n_features), dtype=self.DTYPE)

        fallback_count = 0
        matched_count = 0

        with torch.no_grad():
            Xn_t = torch.from_numpy(X_norm).to(self.device, dtype=torch.float32)

            all_codes: List[np.ndarray] = []
            for table_idx in range(self.n_tables):
                proj = Xn_t @ self._proj_t_list[table_idx].T
                codes = (proj >= 0).cpu().numpy().astype(np.uint8)
                all_codes.append(codes)

        del Xn_t
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        for i in range(n_test):
            total_Y = np.zeros((self.pred_len, self.n_features), dtype=self.DTYPE)
            total_count = 0
            for table_idx in range(self.n_tables):
                hash_code = all_codes[table_idx][i]
                bucket_key = self._hash_code_to_key(hash_code)
                mean_Y, _ = self._find_bucket_mean(
                    hash_code, bucket_key, self.hash_tables[table_idx]
                )
                if mean_Y is not None:
                    total_Y += mean_Y
                    total_count += 1

            if total_count > 0:
                Y_pred[i] = total_Y / total_count
                matched_count += 1
            else:
                fallback_count += 1
                if self.fallback_strategy == 'global_mean':
                    Y_pred[i] = self.global_Y_mean
                else:
                    last_vals = X_flat[i].reshape(self.pred_len, self.n_features)
                    Y_pred[i] = np.full(
                        (self.pred_len, self.n_features),
                        np.mean(last_vals[-1]),
                        dtype=self.DTYPE,
                    )

        if fallback_count > 0:
            logger.warning(f"[LSHSearch] fallback {fallback_count}/{n_test}")
        logger.info(f"[LSHSearch] predict done {Y_pred.shape}, matched={matched_count}")

        if self.n_features == 1:
            Y_pred = Y_pred.squeeze(-1)

        del X_flat, X_norm
        gc.collect()
        return Y_pred

    def get_params(self) -> dict:
        return {
            'model_type': 'LSHSearch',
            'n_hash_funcs': self.n_hash_funcs,
            'n_tables': self.n_tables,
            'hamming_radius': self.hamming_radius,
            'device': str(self.device),
            'n_buckets': sum(len(t) for t in self.hash_tables) if self.is_fitted else 0,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
        }

    def get_bucket_stats(self) -> dict:
        if not self.is_fitted:
            return {}
        stats = {}
        for i, table in enumerate(self.hash_tables):
            bucket_counts = [cnt for _, cnt in table.values()]
            stats[f'table_{i}'] = {
                'n_buckets': len(table),
                'total_samples': sum(bucket_counts),
                'min_bucket': min(bucket_counts) if bucket_counts else 0,
                'max_bucket': max(bucket_counts) if bucket_counts else 0,
                'avg_bucket': float(np.mean(bucket_counts)) if bucket_counts else 0,
            }
        return stats

    def __repr__(self):
        return (
            f"LSHSearch(n_hash={self.n_hash_funcs}, n_tables={self.n_tables}, "
            f"device={self.device})"
        )
