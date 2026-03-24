"""
LSHSearch: 基于局部敏感哈希(Locality Sensitive Hashing)的时序预测基线模型
使用随机投影生成哈希签名，通过哈希桶碰撞实现高效相似检索

内存优化 (v2.0)：
- 哈希桶存储预聚合的 Y 均值（mean_Y, count），不再存储大量索引
- 删除 self.memory_Y 大数组，predict 时直接查均值
- 投影矩阵改用 float32
"""

import numpy as np
from typing import Optional, Tuple, Dict


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
      原来存 {key: [idx1, idx2, ...]}（大量 int 索引），
      现在存 {key: (mean_Y, count)}（每个桶只有 1 个 float32 均值数组 + 1 个 count）
    """

    DTYPE = np.float32   # 全局统一 float32

    def __init__(
        self,
        n_hash_funcs: int = 16,
        n_tables: int = 4,
        hamming_radius: int = 2,
        fallback_strategy: str = 'global_mean',
        random_state: Optional[int] = 42
    ):
        """
        Args:
            n_hash_funcs: 每个哈希表的哈希函数数量（即哈希码位数）
            n_tables: 哈希表数量（增加召回率）
            hamming_radius: 汉明距离容忍半径，超过则不匹配
            fallback_strategy: 桶为空时的备选策略
                             - 'global_mean': 使用全局Y均值
                             - 'random_walk': 使用输入序列最后值
            random_state: 随机种子，保证可复现性
        """
        self.n_hash_funcs = n_hash_funcs
        self.n_tables = n_tables
        self.hamming_radius = hamming_radius
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state

        # 模型状态
        # ================================================
        # 内存优化: List[Dict[str, Tuple[mean_Y, count]]]
        # 不再存储 {key: [idx1, idx2, ...]} 的大索引列表
        # ================================================
        self.hash_tables: list = []
        self.projection_matrices: list = []  # 随机投影矩阵列表（float32）
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        # 维度信息
        self.seq_len: int = 0
        self.n_features: int = 1
        self.pred_len: int = 0

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
        """
        生成随机投影矩阵

        内存优化：直接生成 float32，与数据流全程 dtype 一致
        """
        rng = np.random.RandomState(self.random_state)

        # 高斯随机投影矩阵（float32）
        projection_matrix = rng.randn(self.n_hash_funcs, dim).astype(self.DTYPE)

        # 归一化投影向量（提高数值稳定性）
        norms = np.linalg.norm(projection_matrix, axis=1, keepdims=True)
        projection_matrix = projection_matrix / (norms + 1e-10)

        return projection_matrix

    def _compute_hash_codes(
        self,
        X: np.ndarray,
        projection_matrix: np.ndarray
    ) -> np.ndarray:
        """计算二进制哈希码"""
        # 投影: [n, dim] @ [dim, n_hash] -> [n, n_hash]
        projections = X @ projection_matrix.T
        hash_codes = (projections >= 0).astype(np.uint8)
        return hash_codes

    def _hash_code_to_key(self, hash_code: np.ndarray) -> str:
        """将哈希码数组转换为字符串键"""
        return ''.join(map(str, hash_code.tolist()))

    def _hamming_distance(self, code1: np.ndarray, code2: np.ndarray) -> int:
        """计算两个哈希码之间的汉明距离"""
        return int(np.sum(code1 != code2))

    def _find_candidates_in_bucket(
        self,
        query_code: np.ndarray,
        bucket_key: str,
        table: Dict[str, Tuple[np.ndarray, int]]
    ) -> Tuple[np.ndarray, int]:
        """
        在指定桶中查找候选样本的 Y 均值

        Args:
            query_code: 查询哈希码
            bucket_key: 桶的键
            table: 哈希表，{key: (mean_Y, count)}

        Returns:
            (mean_Y, count) 元组；若桶为空返回 (None, 0)
        """
        # 精确匹配
        if bucket_key in table:
            return table[bucket_key]

        # 汉明半径容忍（检查近似匹配）
        if self.hamming_radius > 0:
            best_key = None
            best_dist = float('inf')
            for stored_key, (mean_Y, cnt) in table.items():
                if stored_key == bucket_key:
                    continue
                dist = self._hamming_distance(query_code,
                    np.array([int(c) for c in stored_key], dtype=np.uint8))
                if dist < best_dist and dist <= self.hamming_radius:
                    best_dist = dist
                    best_key = stored_key
            if best_key is not None:
                return table[best_key]

        return (None, 0)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        构建LSH索引（记忆库）

        内存优化：
        - 遍历一次 X_train/Y_train，在线累加每个哈希桶的 sum_Y 和 count
        - 最终每个桶只存储 mean_Y（float32）和 count（int）
        - 删除 self.memory_Y 大数组
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
            Y_flat = Y_train.reshape(n_samples, -1).astype(self.DTYPE)
        else:
            self.pred_len = Y_train.shape[1]
            Y_flat = Y_train.reshape(n_samples, -1).astype(self.DTYPE)

        y_dim = Y_flat.shape[1]

        # 计算全局 Y 均值（用于 fallback）
        self.global_Y_mean = np.mean(Y_flat, axis=0).astype(self.DTYPE)

        # 实例归一化（必须！LSH对尺度敏感）
        X_norm, _, _ = self._instance_normalize(X_flat)

        # 为每个哈希表生成投影矩阵并构建哈希表
        self.hash_tables = []
        self.projection_matrices = []

        for table_idx in range(self.n_tables):
            # 不同的表使用不同的随机种子
            original_seed = self.random_state
            self.random_state = original_seed + table_idx * 1000

            # 生成投影矩阵
            projection_matrix = self._generate_projection_matrix(X_flat.shape[1])
            self.projection_matrices.append(projection_matrix)

            # 计算哈希码
            hash_codes = self._compute_hash_codes(X_norm, projection_matrix)

            # ================================================
            # 内存优化: 在线累加，不存储索引列表
            # hash_table: {key: (sum_Y, count)}
            # ================================================
            hash_table: Dict[str, Tuple[np.ndarray, int]] = {}
            sum_cache: Dict[str, np.ndarray] = {}
            count_cache: Dict[str, int] = {}

            for i in range(n_samples):
                key = self._hash_code_to_key(hash_codes[i])
                if key not in sum_cache:
                    sum_cache[key] = np.zeros(y_dim, dtype=self.DTYPE)
                    count_cache[key] = 0
                sum_cache[key] += Y_flat[i]
                count_cache[key] += 1

            # 转换为均值形式
            for key in sum_cache:
                cnt = count_cache[key]
                mean_Y = (sum_cache[key] / cnt).astype(self.DTYPE)
                hash_table[key] = (mean_Y, cnt)

            self.hash_tables.append(hash_table)

            # 恢复随机种子
            self.random_state = original_seed

            # 释放中间变量
            del sum_cache, count_cache

        # 释放训练数据（不再需要）
        del Y_flat, X_flat, X_norm, hash_codes
        self.is_fitted = True

        # 打印构建统计
        n_buckets = sum(len(t) for t in self.hash_tables)
        avg_bucket_size = n_samples / max(n_buckets, 1)
        print(f"LSH Index Built (mem-optimized): {self.n_tables} tables, "
              f"{n_buckets} buckets, avg bucket size: {avg_bucket_size:.2f}")

        return self

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        对测试样本进行预测

        内存优化：
        - 精确匹配：直接查表取均值，O(1)
        - 聚合：多表结果取平均（而非收集索引后重复索引访问 memory_Y）
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]

        # 展平
        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        # 实例归一化
        X_norm, _, _ = self._instance_normalize(X_flat)

        y_dim = self.pred_len * self.n_features
        Y_pred = np.zeros((n_test, y_dim), dtype=self.DTYPE)
        fallback_count = 0
        matched_count = 0

        for i in range(n_test):
            # 收集所有表对该样本的候选均值
            total_Y = np.zeros(y_dim, dtype=self.DTYPE)
            total_count = 0

            for table_idx in range(self.n_tables):
                projection_matrix = self.projection_matrices[table_idx]
                hash_table = self.hash_tables[table_idx]

                # 计算查询哈希码
                hash_code = self._compute_hash_codes(X_norm[i:i+1], projection_matrix)[0]
                bucket_key = self._hash_code_to_key(hash_code)

                # 从桶中取均值
                mean_Y, cnt = self._find_candidates_in_bucket(
                    hash_code, bucket_key, hash_table
                )
                if mean_Y is not None:
                    total_Y += mean_Y
                    total_count += 1

            # 聚合多表结果
            if total_count > 0:
                Y_pred[i] = total_Y / total_count
                matched_count += 1
            else:
                # Fallback策略
                fallback_count += 1
                if self.fallback_strategy == 'global_mean':
                    Y_pred[i] = self.global_Y_mean
                else:  # random_walk
                    last_val = np.mean(X_flat[i].reshape(-1, self.n_features)[-1])
                    Y_pred[i] = np.full(y_dim, last_val, dtype=self.DTYPE)

        if fallback_count > 0:
            print(f"  Warning: {fallback_count}/{n_test} samples used fallback strategy")
        if matched_count > 0:
            print(f"  Matched: {matched_count}/{n_test} samples via LSH lookup")

        # 恢复原始形状
        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

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
                'avg_bucket': np.mean(bucket_counts) if bucket_counts else 0,
            }

        return stats

    def __repr__(self):
        return (f"LSHSearch(n_hash={self.n_hash_funcs}, n_tables={self.n_tables}, "
                f"hamming_radius={self.hamming_radius})")
