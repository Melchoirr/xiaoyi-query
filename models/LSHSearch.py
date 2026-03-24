"""
LSHSearch: 基于局部敏感哈希(Locality Sensitive Hashing)的时序预测基线模型
使用随机投影生成哈希签名，通过哈希桶碰撞实现高效相似检索

Bug 修复 (v2.1):
- __init__ 末尾追加 **kwargs，兼容所有外部传入参数，防止 TypeError
- fit(): sum_cache 按 (pred_len, n_features) 形状初始化，保证多变量不坍缩
- predict(): mean_Y 累加后 reshape 回 (pred_len, n_features)，解决 broadcast 报错
- 引入 logging + tqdm，提供进度可见性
"""

import numpy as np
import gc
import logging
from typing import Optional, Tuple, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LSHSearch:
    """
    基于局部敏感哈希的时序预测基线模型

    使用Random Projection LSH (E2LSH)变体：
    - 生成k个随机投影向量
    - 每个投影将序列映射到 {-1, +1}
    - k个符号构成k位哈希码
    - 相似序列以高概率产生相同或相近的哈希码

    内存优化：
    - self.hash_tables: List[Dict[str, Tuple[mean_Y, count]]]
      每个桶存预聚合的 (mean_Y_flat, count)，不再存大量索引
    """

    DTYPE = np.float32   # 全局统一 float32

    def __init__(
        self,
        # ── 兼容 run.py 的参数名 ──
        n_hash_funcs: int = 16,
        n_tables: int = 4,
        hamming_radius: int = 2,
        fallback_strategy: str = 'global_mean',
        random_state: Optional[int] = 42,
        # ── 基础维度参数（预留）──
        seq_len: int = 0,
        pred_len: int = 0,
        n_features: int = 1,
        # ── 安全吸收未声明参数，防止 TypeError ──
        **kwargs
    ):
        self.n_hash_funcs = n_hash_funcs
        self.n_tables = n_tables
        self.hamming_radius = hamming_radius
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state

        # 模型状态
        self.hash_tables: list = []
        self.projection_matrices: list = []
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        # 维度信息
        self.seq_len: int = seq_len
        self.n_features: int = n_features
        self.pred_len: int = pred_len

        logger.debug(f"[LSHSearch] init: n_hash={n_hash_funcs}, "
                     f"n_tables={n_tables}, hamming={hamming_radius}")

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """实例级Z-Score归一化"""
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
        """生成随机投影矩阵（float32）"""
        rng = np.random.RandomState(self.random_state)
        projection_matrix = rng.randn(self.n_hash_funcs, dim).astype(self.DTYPE)
        norms = np.linalg.norm(projection_matrix, axis=1, keepdims=True)
        projection_matrix = projection_matrix / (norms + 1e-10)
        return projection_matrix

    def _compute_hash_codes(
        self,
        X: np.ndarray,
        projection_matrix: np.ndarray
    ) -> np.ndarray:
        """计算二进制哈希码"""
        projections = X @ projection_matrix.T
        hash_codes = (projections >= 0).astype(np.uint8)
        return hash_codes

    def _hash_code_to_key(self, hash_code: np.ndarray) -> str:
        """将哈希码数组转换为字符串键"""
        return ''.join(map(str, hash_code.tolist()))

    def _hamming_distance(self, code1: np.ndarray, code2: np.ndarray) -> int:
        """计算两个哈希码之间的汉明距离"""
        return int(np.sum(code1 != code2))

    def _find_bucket_mean(
        self,
        query_code: np.ndarray,
        bucket_key: str,
        table: Dict[str, Tuple[np.ndarray, int]]
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        在指定桶中查找 Y 均值

        Args:
            query_code: 查询哈希码
            bucket_key: 桶的键
            table: 哈希表，{key: (mean_Y_flat, count)}

        Returns:
            (mean_Y_flat, count) 元组；若桶为空返回 (None, 0)
        """
        # 精确匹配
        if bucket_key in table:
            return table[bucket_key]

        # 汉明半径容忍
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
        """
        构建LSH索引（记忆库）

        Bug 修复 (v2.1):
        - sum_cache 形状必须为 (pred_len, n_features)，不能是 (y_dim,)
          否则多变量时 predict 端无法正确 reshape 回 (pred_len, n_features)
        """
        n_samples = X_train.shape[0]

        # 保存维度信息
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
            Y_original = Y_train.astype(self.DTYPE)       # (n, pred_len, n_feat)
        else:
            self.pred_len = Y_train.shape[1]
            self.n_features = 1
            Y_original = Y_train.astype(self.DTYPE)

        # ── 多变量关键修复 ──
        # sum_cache 形状为 (pred_len, n_features)，而非扁平的 y_dim
        # 这样均值可以直接 reshape 回 (pred_len, n_features)
        sum_cache: Dict[str, np.ndarray] = {}
        count_cache: Dict[str, int] = {}

        logger.info(f"[LSHSearch] fit start: X={X_flat.shape}, Y={Y_original.shape}, "
                    f"pred_len={self.pred_len}, n_features={self.n_features}")

        # 实例归一化
        X_norm, _, _ = self._instance_normalize(X_flat)

        for table_idx in range(self.n_tables):
            original_seed = self.random_state
            self.random_state = original_seed + table_idx * 1000

            # 生成投影矩阵
            projection_matrix = self._generate_projection_matrix(X_flat.shape[1])
            self.projection_matrices.append(projection_matrix)

            # 计算哈希码
            hash_codes = self._compute_hash_codes(X_norm, projection_matrix)

            # 构建哈希表：在线累加 Y（保持原始 (pred_len, n_features) 形状）
            hash_table: Dict[str, Tuple[np.ndarray, int]] = {}

            for i in range(n_samples):
                key = self._hash_code_to_key(hash_codes[i])
                if key not in sum_cache:
                    sum_cache[key] = np.zeros(
                        (self.pred_len, self.n_features), dtype=self.DTYPE)
                    count_cache[key] = 0
                # Y_original[i] shape: (pred_len, n_features)
                sum_cache[key] += Y_original[i]
                count_cache[key] += 1

            # 转换为均值形式
            for key in sum_cache:
                cnt = count_cache[key]
                mean_Y = (sum_cache[key] / cnt).astype(self.DTYPE)
                hash_table[key] = (mean_Y, cnt)

            self.hash_tables.append(hash_table)

            self.random_state = original_seed
            logger.info(f"[LSHSearch] table {table_idx}: {len(hash_table)} buckets built")

        # 全局 Y 均值（用于 fallback）
        self.global_Y_mean = np.mean(Y_original, axis=0).astype(self.DTYPE)

        # 释放中间变量
        del sum_cache, count_cache, Y_original, X_flat, X_norm, hash_codes
        gc.collect()
        self.is_fitted = True

        n_buckets = sum(len(t) for t in self.hash_tables)
        avg_bucket_size = n_samples / max(n_buckets, 1)
        logger.info(f"[LSHSearch] fitted: {self.n_tables} tables, {n_buckets} buckets, "
                     f"avg bucket size: {avg_bucket_size:.2f}")

        return self

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        对测试样本进行预测

        Bug 修复 (v2.1):
        - mean_Y 直接是 (pred_len, n_features) 形状，无需 flatten → reshape 二次操作
        - 多表均值聚合后直接 reshape，避免 broadcast 错误
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]

        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)

        # 预分配结果数组，形状 (n_test, pred_len, n_features)
        Y_pred = np.zeros((n_test, self.pred_len, self.n_features), dtype=self.DTYPE)

        fallback_count = 0
        matched_count = 0

        for i in tqdm(range(n_test), desc="[LSHSearch] Predicting", unit="sample"):
            # 收集所有表对该样本的候选均值
            total_Y = np.zeros((self.pred_len, self.n_features), dtype=self.DTYPE)
            total_count = 0

            for table_idx in range(self.n_tables):
                projection_matrix = self.projection_matrices[table_idx]
                hash_table = self.hash_tables[table_idx]

                hash_code = self._compute_hash_codes(X_norm[i:i+1], projection_matrix)[0]
                bucket_key = self._hash_code_to_key(hash_code)

                mean_Y, cnt = self._find_bucket_mean(hash_code, bucket_key, hash_table)
                if mean_Y is not None:
                    total_Y += mean_Y
                    total_count += 1

            # 聚合多表结果
            if total_count > 0:
                Y_pred[i] = total_Y / total_count
                matched_count += 1
            else:
                fallback_count += 1
                if self.fallback_strategy == 'global_mean':
                    Y_pred[i] = self.global_Y_mean
                else:  # random_walk
                    last_vals = X_flat[i].reshape(self.pred_len, self.n_features)
                    Y_pred[i] = np.full(
                        (self.pred_len, self.n_features),
                        np.mean(last_vals[-1]),
                        dtype=self.DTYPE
                    )

        if fallback_count > 0:
            logger.warning(f"[LSHSearch] {fallback_count}/{n_test} samples used fallback")
        if matched_count > 0:
            logger.info(f"[LSHSearch] {matched_count}/{n_test} samples matched via LSH")

        # 恢复形状（单变量时降为 2D）
        if self.n_features == 1:
            Y_pred = Y_pred.squeeze(-1)  # (n, pred_len, 1) -> (n, pred_len)

        del X_flat, X_norm
        gc.collect()

        logger.info(f"[LSHSearch] predict done: {Y_pred.shape}")
        return Y_pred

    def get_params(self) -> dict:
        """获取模型参数"""
        return {
            'model_type': 'LSHSearch',
            'n_hash_funcs': self.n_hash_funcs,
            'n_tables': self.n_tables,
            'hamming_radius': self.hamming_radius,
            'fallback_strategy': self.fallback_strategy,
            'n_buckets': sum(len(t) for t in self.hash_tables) if self.is_fitted else 0,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features
        }

    def get_bucket_stats(self) -> dict:
        """获取哈希表统计信息"""
        if not self.is_fitted:
            return {}

        stats = {}
        for i, table in enumerate(self.hash_tables):
            bucket_counts = [cnt for _, cnt in table.values()]
            total = sum(bucket_counts)
            stats[f'table_{i}'] = {
                'n_buckets': len(table),
                'total_samples': total,
                'min_bucket': min(bucket_counts) if bucket_counts else 0,
                'max_bucket': max(bucket_counts) if bucket_counts else 0,
                'avg_bucket': float(np.mean(bucket_counts)) if bucket_counts else 0,
            }
        return stats

    def __repr__(self):
        return (f"LSHSearch(n_hash={self.n_hash_funcs}, n_tables={self.n_tables}, "
                f"hamming_radius={self.hamming_radius})")
