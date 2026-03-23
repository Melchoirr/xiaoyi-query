"""
LSHSearch: 基于局部敏感哈希(Locality Sensitive Hashing)的时序预测基线模型
使用随机投影生成哈希签名，通过哈希桶碰撞实现高效相似检索

核心思想：
- 对每个序列进行Instance Normalization（实例级Z-Score）
- 使用多组随机超平面投影生成二进制哈希码
- 将哈希码作为键存储历史(Y)值到哈希表
- 查询时计算哈希码，查找碰撞桶进行预测
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
    """

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
        self.hash_tables: list = []  # 多个哈希表
        self.projection_matrices: list = []  # 随机投影矩阵列表
        self.memory_Y: Optional[np.ndarray] = None
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        # 维度信息
        self.seq_len: int = 0
        self.n_features: int = 1
        self.pred_len: int = 0

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        实例级Z-Score归一化

        对每个序列独立进行归一化，保留波形形状特征

        Args:
            X: 输入序列 [n_samples, seq_len] 或 [n_samples, seq_len, n_features]

        Returns:
            X_norm: 归一化后的序列
            X_mean: 每个样本的均值 [n_samples, 1] 或 [n_samples, 1, n_features]
            X_std: 每个样本的标准差 [n_samples, 1] 或 [n_samples, 1, n_features]
        """
        if X.ndim == 3:
            # [n, T, d] -> 计算每个样本的均值和标准差
            X_mean = np.mean(X, axis=1, keepdims=True)  # [n, 1, d]
            X_std = np.std(X, axis=1, keepdims=True)  # [n, 1, d]
        else:
            # [n, T] -> 计算每个样本的均值和标准差
            X_mean = np.mean(X, axis=1, keepdims=True)  # [n, 1]
            X_std = np.std(X, axis=1, keepdims=True)  # [n, 1]

        # 避免除零
        X_std = np.clip(X_std, 1e-8, None)

        X_norm = (X - X_mean) / X_std

        return X_norm, X_mean, X_std

    def _generate_projection_matrix(self, dim: int) -> np.ndarray:
        """
        生成随机投影矩阵

        使用高斯随机投影: R_{kxd}, R[i,j] ~ N(0, 1)

        Args:
            dim: 投影目标维度 (即每个序列的展平长度)

        Returns:
            projection_matrix: shape [n_hash_funcs, dim]
        """
        rng = np.random.RandomState(self.random_state)

        # 高斯随机投影矩阵
        projection_matrix = rng.randn(self.n_hash_funcs, dim).astype(np.float64)

        # 归一化投影向量（提高数值稳定性）
        norms = np.linalg.norm(projection_matrix, axis=1, keepdims=True)
        projection_matrix = projection_matrix / (norms + 1e-10)

        return projection_matrix

    def _compute_hash_codes(
        self,
        X: np.ndarray,
        projection_matrix: np.ndarray
    ) -> np.ndarray:
        """
        计算二进制哈希码

        符号函数: h(x) = sign(x · r), 其中 r 是随机投影向量
        sign(正数) = 1, sign(负数) = 0

        Args:
            X: 展平后的序列矩阵 [n_samples, dim]
            projection_matrix: 投影矩阵 [n_hash_funcs, dim]

        Returns:
            hash_codes: 二进制哈希码 [n_samples, n_hash_funcs], dtype=uint8
        """
        # 投影: [n, dim] @ [dim, n_hash] -> [n, n_hash]
        projections = X @ projection_matrix.T

        # 符号函数: 正数->1, 负数->0, 0->1 (约定)
        hash_codes = (projections >= 0).astype(np.uint8)

        return hash_codes

    def _hash_code_to_key(self, hash_code: np.ndarray) -> str:
        """
        将哈希码数组转换为字符串键

        Args:
            hash_code: 二进制数组, shape [n_hash_funcs]

        Returns:
            key: 字符串形式的哈希码, 如 "01011010"
        """
        return ''.join(map(str, hash_code.tolist()))

    def _hamming_distance(self, code1: np.ndarray, code2: np.ndarray) -> int:
        """
        计算两个哈希码之间的汉明距离

        Args:
            code1, code2: 二进制数组

        Returns:
            汉明距离（不同位的数量）
        """
        return int(np.sum(code1 != code2))

    def _find_candidates_in_bucket(
        self,
        query_code: np.ndarray,
        bucket_key: str,
        table: Dict[str, list]
    ) -> list:
        """
        在指定桶中查找汉明距离在容忍范围内的候选样本

        Args:
            query_code: 查询哈希码
            bucket_key: 桶的键
            table: 哈希表

        Returns:
            candidate_indices: 候选样本索引列表
        """
        candidates = []

        # 首先检查精确匹配的键
        if bucket_key in table:
            candidates.extend(table[bucket_key])

        # 如果设置了汉明半径容忍，检查近似匹配
        if self.hamming_radius > 0:
            for stored_key, indices in table.items():
                if stored_key == bucket_key:
                    continue
                stored_code = np.array([int(c) for c in stored_key], dtype=np.uint8)
                dist = self._hamming_distance(query_code, stored_code)
                if dist <= self.hamming_radius:
                    candidates.extend(indices)

        return list(set(candidates))

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        构建LSH索引（记忆库）

        Args:
            X_train: 输入序列数组
                   shape: [n_samples, seq_len, n_features] 或 [n_samples, seq_len]
            Y_train: 目标序列数组
                   shape: [n_samples, pred_len, n_features] 或 [n_samples, pred_len]
        """
        n_samples = X_train.shape[0]

        # 保存维度信息
        if X_train.ndim == 3:
            self.seq_len = X_train.shape[1]
            self.n_features = X_train.shape[2]
            X_flat = X_train.reshape(n_samples, -1)  # [n, seq_len * n_features]
        else:
            self.seq_len = X_train.shape[1]
            self.n_features = 1
            X_flat = X_train.reshape(n_samples, -1)

        if Y_train.ndim == 3:
            self.pred_len = Y_train.shape[1]
            self.memory_Y = Y_train.reshape(n_samples, -1)  # [n, pred_len * n_features]
        else:
            self.pred_len = Y_train.shape[1]
            self.memory_Y = Y_train.reshape(n_samples, -1)

        # 计算全局Y均值（用于fallback）
        self.global_Y_mean = np.mean(self.memory_Y, axis=0)

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

            # 构建哈希表: {hash_key: [sample_indices]}
            hash_table = {}
            for i in range(n_samples):
                key = self._hash_code_to_key(hash_codes[i])
                if key not in hash_table:
                    hash_table[key] = []
                hash_table[key].append(i)

            self.hash_tables.append(hash_table)

            # 恢复随机种子
            self.random_state = original_seed

        self.is_fitted = True

        # 打印构建统计
        n_buckets = sum(len(t) for t in self.hash_tables)
        avg_bucket_size = n_samples / max(n_buckets, 1)
        print(f"LSH Index Built: {self.n_tables} tables, {n_buckets} buckets, "
              f"avg bucket size: {avg_bucket_size:.2f}")

        return self

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        对测试样本进行预测

        Args:
            X_test: 测试输入序列
                   shape: [n_test, seq_len, n_features] 或 [n_test, seq_len]
            top_k: 最多聚合的样本数量

        Returns:
            Y_pred: 预测输出
                   shape: [n_test, pred_len, n_features] 或 [n_test, pred_len]
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]

        # 展平
        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1)
        else:
            X_flat = X_test.reshape(n_test, -1)

        # 实例归一化
        X_norm, _, _ = self._instance_normalize(X_flat)

        # 收集所有表的候选样本
        all_candidates = []  # list of lists

        for table_idx in range(self.n_tables):
            projection_matrix = self.projection_matrices[table_idx]
            hash_table = self.hash_tables[table_idx]

            # 计算哈希码
            hash_codes = self._compute_hash_codes(X_norm, projection_matrix)

            # 查找候选
            table_candidates = []
            for i in range(n_test):
                key = self._hash_code_to_key(hash_codes[i])
                candidates = self._find_candidates_in_bucket(
                    hash_codes[i], key, hash_table
                )
                table_candidates.append(candidates)

            all_candidates.append(table_candidates)

        # 合并所有表的候选（去重并限制数量）
        final_candidates = []
        for i in range(n_test):
            merged = set()
            for table_idx in range(self.n_tables):
                merged.update(all_candidates[table_idx][i])
            # 限制数量
            merged = list(merged)[:top_k]
            final_candidates.append(merged)

        # 聚合预测
        Y_pred = np.zeros((n_test, self.pred_len * self.n_features), dtype=np.float64)

        fallback_count = 0
        for i in range(n_test):
            candidates = final_candidates[i]
            if len(candidates) > 0:
                # 简单平均（LSH不保持原始距离度量）
                candidate_Y = self.memory_Y[candidates]
                Y_pred[i] = np.mean(candidate_Y, axis=0)
            else:
                # Fallback策略
                fallback_count += 1
                if self.fallback_strategy == 'global_mean':
                    Y_pred[i] = self.global_Y_mean
                else:  # random_walk
                    # 使用输入序列最后值的均值作为预测
                    last_values = X_flat[i].reshape(-1, self.n_features)
                    Y_pred[i] = np.tile(np.mean(last_values[-1]), self.pred_len)

        if fallback_count > 0:
            print(f"  Warning: {fallback_count}/{n_test} samples used fallback strategy")

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
            'n_memory': self.memory_Y.shape[0] if self.is_fitted else 0,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features
        }

    def get_bucket_stats(self) -> dict:
        """
        获取哈希表统计信息（用于分析）

        Returns:
            包含各表统计信息的字典
        """
        if not self.is_fitted:
            return {}

        stats = {}
        for i, table in enumerate(self.hash_tables):
            bucket_sizes = [len(indices) for indices in table.values()]
            stats[f'table_{i}'] = {
                'n_buckets': len(table),
                'min_bucket': min(bucket_sizes) if bucket_sizes else 0,
                'max_bucket': max(bucket_sizes) if bucket_sizes else 0,
                'avg_bucket': np.mean(bucket_sizes) if bucket_sizes else 0,
            }

        return stats

    def __repr__(self):
        return (f"LSHSearch(n_hash={self.n_hash_funcs}, n_tables={self.n_tables}, "
                f"hamming_radius={self.hamming_radius})")
