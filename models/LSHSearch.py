"""
LSHSearch: locality-sensitive hashing baseline for time-series retrieval.

This implementation uses a two-stage pipeline:
1. Hash each normalized input into multiple tables and retrieve candidate indices.
2. Rerank only the retrieved candidates with exact distances.

The design avoids the previous full-table scan on Hamming misses, which turned
the hot path into a Python-bound lookup workload.
"""

import gc
import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

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
        candidate_cap_per_table: int = 256,
        candidate_cap_total: int = 1024,
        weighted: bool = False,
        **kwargs
    ):
        self.n_hash_funcs = n_hash_funcs
        self.n_tables = n_tables
        self.hamming_radius = max(0, int(hamming_radius))
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.candidate_cap_per_table = max(1, int(candidate_cap_per_table))
        self.candidate_cap_total = max(self.candidate_cap_per_table, int(candidate_cap_total))
        self.weighted = weighted

        self.hash_tables: List[Dict[int, np.ndarray]] = []
        self.projection_matrices: List[np.ndarray] = []
        self._proj_t_list: List[torch.Tensor] = []
        self._probe_masks: List[int] = []

        self.memory_X: Optional[np.ndarray] = None
        self.memory_Y: Optional[np.ndarray] = None
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted = False
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len

    # ─────────────────────────────────────────────────────────
    #  Normalization
    # ─────────────────────────────────────────────────────────

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_std = np.clip(X_std, 1e-8, None)
        X_norm = (X - X_mean) / X_std
        return X_norm.astype(self.DTYPE), X_mean.astype(self.DTYPE), X_std.astype(self.DTYPE)

    # ─────────────────────────────────────────────────────────
    #  Projection matrix
    # ─────────────────────────────────────────────────────────

    def _generate_projection_matrix(self, dim: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        projection_matrix = rng.randn(self.n_hash_funcs, dim).astype(self.DTYPE)
        norms = np.linalg.norm(projection_matrix, axis=1, keepdims=True)
        projection_matrix = projection_matrix / (norms + 1e-10)
        return projection_matrix

    # ─────────────────────────────────────────────────────────
    #  uint64 hash-code packing
    # ─────────────────────────────────────────────────────────

    def _pack_hash_codes(self, hash_codes: np.ndarray) -> np.ndarray:
        """
        Pack binary hash codes into uint64 keys.
        This keeps the lookup path numeric and avoids string churn.
        """
        if self.n_hash_funcs > 63:
            raise ValueError("LSHSearch currently supports n_hash_funcs <= 63")

        weights = (1 << np.arange(self.n_hash_funcs, dtype=np.uint64)).reshape(1, -1)
        packed = (hash_codes.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)
        return packed

    # ─────────────────────────────────────────────────────────
    #  Multi-probe masks (Hamming radius enumeration)
    # ─────────────────────────────────────────────────────────

    def _build_probe_masks(self) -> List[int]:
        if self.hamming_radius <= 0:
            return []

        masks: List[int] = []
        bit_positions = list(range(self.n_hash_funcs))
        max_radius = min(self.hamming_radius, self.n_hash_funcs)

        for radius in range(1, max_radius + 1):
            for combo in combinations(bit_positions, radius):
                mask = 0
                for pos in combo:
                    mask |= (1 << pos)
                masks.append(mask)
        return masks

    # ─────────────────────────────────────────────────────────
    #  Candidate retrieval
    # ─────────────────────────────────────────────────────────

    def _limit_indices(self, indices: np.ndarray, cap: int) -> np.ndarray:
        if indices.size <= cap:
            return indices
        step_positions = np.linspace(0, indices.size - 1, num=cap, dtype=np.int64)
        return indices[step_positions]

    def _retrieve_candidates(self, query_key: int, table: Dict[int, np.ndarray]) -> List[np.ndarray]:
        buckets: List[np.ndarray] = []

        exact = table.get(query_key)
        if exact is not None:
            buckets.append(self._limit_indices(exact, self.candidate_cap_per_table))

        if self.hamming_radius <= 0:
            return buckets

        for mask in self._probe_masks:
            nearby = table.get(query_key ^ mask)
            if nearby is not None:
                buckets.append(self._limit_indices(nearby, self.candidate_cap_per_table))
        return buckets

    # ─────────────────────────────────────────────────────────
    #  Exact reranking on candidate set
    # ─────────────────────────────────────────────────────────

    def _rerank_candidates(self, query_vec: np.ndarray, candidate_idx: np.ndarray, top_k: int) -> np.ndarray:
        candidate_X = self.memory_X[candidate_idx]
        diff = candidate_X - query_vec.reshape(1, -1)
        dist = np.einsum('ij,ij->i', diff, diff, optimize=True)

        k = min(top_k, candidate_idx.size)
        if k <= 0:
            return self.global_Y_mean.copy()

        top_pos = np.argpartition(dist, k - 1)[:k]
        selected_y = self.memory_Y[candidate_idx[top_pos]]

        if self.weighted:
            selected_dist = np.maximum(dist[top_pos], 1e-10)
            weight = 1.0 / selected_dist
            weight = weight / np.sum(weight)
            return np.tensordot(weight.astype(self.DTYPE), selected_y, axes=(0, 0)).astype(self.DTYPE)

        return selected_y.mean(axis=0).astype(self.DTYPE)

    # ─────────────────────────────────────────────────────────
    #  fit()
    # ─────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        n_samples = X_train.shape[0]

        if X_train.ndim == 3:
            n_samples, seq_len, n_features = X_train.shape
            X_flat = X_train.reshape(n_samples, -1).astype(self.DTYPE)
            Y_original = Y_train.astype(self.DTYPE)
        else:
            X_flat = X_train.astype(self.DTYPE)
            n_features = 1
            Y_original = Y_train.astype(self.DTYPE)

        if self.n_features <= 0:
            self.n_features = 1
            Y_original = Y_train.astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)
        self.memory_X = X_norm
        self.memory_Y = Y_original
        self.global_Y_mean = np.mean(Y_original, axis=0).astype(self.DTYPE)
        self._probe_masks = self._build_probe_masks()

        self.hash_tables = []
        self.projection_matrices = []
        self._proj_t_list = []

        Xn_t = torch.from_numpy(X_norm).to(self.device, dtype=torch.float32)

        for table_idx in range(self.n_tables):
            seed = (self.random_state or 0) + table_idx * 1000
            projection_matrix = self._generate_projection_matrix(X_flat.shape[1], seed)
            self.projection_matrices.append(projection_matrix)

            proj_t = torch.from_numpy(projection_matrix).to(self.device, dtype=torch.float32)
            self._proj_t_list.append(proj_t)

            with torch.no_grad():
                proj = Xn_t @ proj_t.T
                hash_codes = (proj >= 0).cpu().numpy().astype(np.uint8)

            packed_keys = self._pack_hash_codes(hash_codes)
            bucket_lists: Dict[int, List[int]] = {}
            for idx, key in enumerate(packed_keys):
                bucket_lists.setdefault(int(key), []).append(idx)

            hash_table: Dict[int, np.ndarray] = {}
            for key, values in bucket_lists.items():
                hash_table[key] = np.asarray(values, dtype=np.int32)
            self.hash_tables.append(hash_table)

        del X_flat, Xn_t
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.is_fitted = True
        n_buckets = sum(len(t) for t in self.hash_tables)
        logger.info(
            f"[LSHSearch] fitted device={self.device}, tables={self.n_tables}, "
            f"buckets={n_buckets}, probe_masks={len(self._probe_masks)}"
        )
        return self

    # ─────────────────────────────────────────────────────────
    #  predict()
    # ─────────────────────────────────────────────────────────

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        if not self.is_fitted or self.memory_X is None or self.memory_Y is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]

        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            X_flat = X_test.astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)
        Y_pred = np.zeros((n_test, self.pred_len, self.n_features), dtype=self.DTYPE)

        Xn_t = torch.from_numpy(X_norm).to(self.device, dtype=torch.float32)
        all_packed_keys: List[np.ndarray] = []

        with torch.no_grad():
            for table_idx in range(self.n_tables):
                proj = Xn_t @ self._proj_t_list[table_idx].T
                hash_codes = (proj >= 0).cpu().numpy().astype(np.uint8)
                all_packed_keys.append(self._pack_hash_codes(hash_codes))

        del Xn_t
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        matched_count = 0
        fallback_count = 0
        candidate_sizes: List[int] = []

        for i in range(n_test):
            candidate_chunks: List[np.ndarray] = []

            for table_idx in range(self.n_tables):
                query_key = int(all_packed_keys[table_idx][i])
                candidate_chunks.extend(
                    self._retrieve_candidates(query_key, self.hash_tables[table_idx])
                )

            if candidate_chunks:
                candidate_idx = np.unique(np.concatenate(candidate_chunks))
                if candidate_idx.size > self.candidate_cap_total:
                    candidate_idx = self._limit_indices(candidate_idx, self.candidate_cap_total)

                Y_pred[i] = self._rerank_candidates(X_norm[i], candidate_idx, top_k=top_k)
                matched_count += 1
                candidate_sizes.append(int(candidate_idx.size))
            else:
                fallback_count += 1
                if self.fallback_strategy == 'global_mean':
                    Y_pred[i] = self.global_Y_mean
                else:
                    last_vals = X_flat[i].reshape(self.seq_len, self.n_features)
                    fill_value = float(np.mean(last_vals[-1]))
                    Y_pred[i] = np.full(
                        (self.pred_len, self.n_features),
                        fill_value,
                        dtype=self.DTYPE,
                    )

        if fallback_count > 0:
            logger.warning(f"[LSHSearch] fallback {fallback_count}/{n_test}")

        avg_candidates = float(np.mean(candidate_sizes)) if candidate_sizes else 0.0
        max_candidates = int(np.max(candidate_sizes)) if candidate_sizes else 0
        logger.info(
            f"[LSHSearch] predict done {Y_pred.shape}, matched={matched_count}, "
            f"avg_candidates={avg_candidates:.1f}, max_candidates={max_candidates}"
        )

        if self.n_features == 1:
            Y_pred = Y_pred.squeeze(-1)

        del X_flat, X_norm
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return Y_pred

    # ─────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            'n_hash_funcs': self.n_hash_funcs,
            'n_tables': self.n_tables,
            'hamming_radius': self.hamming_radius,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'candidate_cap_per_table': self.candidate_cap_per_table,
            'candidate_cap_total': self.candidate_cap_total,
        }

    def get_bucket_stats(self) -> dict:
        if not self.is_fitted:
            return {}
        stats = {}
        for i, table in enumerate(self.hash_tables):
            bucket_counts = [len(idx) for idx in table.values()]
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
            f"LSHSearch(n_hash_funcs={self.n_hash_funcs}, n_tables={self.n_tables}, "
            f"hamming_radius={self.hamming_radius}, weighted={self.weighted}, "
            f"cap_per_table={self.candidate_cap_per_table}, cap_total={self.candidate_cap_total})"
        )
