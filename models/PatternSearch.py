"""
PatternSearch: 基于记忆检索的时序预测基线模型
使用 k-NN / KD-Tree 近邻检索算法寻找相似波形并进行预测
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class PatternSearch:
    """
    基于记忆检索的时序预测基线模型

    核心思想：
    - 将历史输入序列 X 作为记忆库
    - 对每个测试样本，通过 k-NN 找到最相似的 K 个历史序列
    - 使用对应历史序列的真实输出 Y 进行加权融合预测
    """

    def __init__(self, k: int = 5, weighted: bool = True, algorithm: str = 'kd_tree'):
        """
        Args:
            k: 近邻数量，默认为5
            weighted: 是否使用逆距离加权，True为逆距离加权，False为简单平均
            algorithm: 近邻搜索算法，'kd_tree', 'ball_tree', 'brute', 'auto'
        """
        self.k = k
        self.weighted = weighted
        self.algorithm = algorithm
        self.nn_model = None
        self.memory_X = None
        self.memory_Y = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        构建记忆库，使用训练集的 (X, Y) 样本

        Args:
            X_train: 输入序列数组，shape: [n_samples, seq_len, n_features]
                    或 shape: [n_samples, seq_len] (单变量)
            Y_train: 目标序列数组，shape: [n_samples, pred_len, n_features]
                    或 shape: [n_samples, pred_len] (单变量)

        Note:
            对于多变量情况，我们通常将多维序列展平成一维向量进行相似度计算
        """
        n_samples = X_train.shape[0]

        # 展平输入序列：将 [n, seq_len, d] -> [n, seq_len * d]
        if X_train.ndim == 3:
            self.seq_len = X_train.shape[1]
            self.n_features = X_train.shape[2]
            self.memory_X = X_train.reshape(n_samples, -1)
        else:
            self.seq_len = X_train.shape[1]
            self.n_features = 1
            self.memory_X = X_train.reshape(n_samples, -1)

        # 保存展平后的目标序列
        if Y_train.ndim == 3:
            self.pred_len = Y_train.shape[1]
            self.memory_Y = Y_train.reshape(n_samples, -1)
        else:
            self.pred_len = Y_train.shape[1]
            self.memory_Y = Y_train.reshape(n_samples, -1)

        # 使用 sklearn NearestNeighbors 构建 KD-Tree 索引
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.k, n_samples),
            algorithm=self.algorithm,
            metric='euclidean',
            n_jobs=-1
        )
        self.nn_model.fit(self.memory_X)
        self.is_fitted = True

        return self

    def predict(self, X_test: np.ndarray, top_k: int = None) -> np.ndarray:
        """
        对测试样本进行预测

        Args:
            X_test: 测试输入序列，shape: [n_test_samples, seq_len, n_features]
                   或 shape: [n_test_samples, seq_len] (单变量)
            top_k: 覆盖默认的k值，可指定不同的k进行预测

        Returns:
            Y_pred: 预测输出，shape: [n_test_samples, pred_len, n_features]
                   或 shape: [n_test_samples, pred_len] (单变量)
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        k = top_k if top_k is not None else self.k
        k = min(k, self.memory_X.shape[0])

        # 展平测试输入
        if X_test.ndim == 3:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1)
        else:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1)

        # 在记忆库中搜索 k 个最近邻
        distances, indices = self.nn_model.kneighbors(X_flat, n_neighbors=k)

        # 检索对应的 Y 值
        neighbor_Y = self.memory_Y[indices]  # shape: [n_test, k, pred_len * n_features]

        # 融合策略：计算预测
        if self.weighted:
            # 逆距离加权平均
            # 避免距离为0时权重无穷大，添加小常数
            distances = np.clip(distances, 1e-10, None)
            weights = 1.0 / distances  # shape: [n_test, k]
            weights = weights / weights.sum(axis=1, keepdims=True)  # 归一化

            # 加权平均: [n_test, k, dim] -> [n_test, dim]
            Y_pred = np.sum(neighbor_Y * weights[:, :, np.newaxis], axis=1)
        else:
            # 简单平均
            Y_pred = np.mean(neighbor_Y, axis=1)

        # 恢复原始形状
        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        return Y_pred

    def get_neighbors(self, X_query: np.ndarray, top_k: int = None):
        """
        获取查询样本的 k 个最近邻及其距离（用于分析）

        Args:
            X_query: 查询序列，shape: [n_queries, seq_len, n_features]
            top_k: 近邻数量

        Returns:
            distances: 距离数组，shape: [n_queries, top_k]
            indices: 索引数组，shape: [n_queries, top_k]
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        k = top_k if top_k is not None else self.k
        k = min(k, self.memory_X.shape[0])

        if X_query.ndim == 3:
            n_queries = X_query.shape[0]
            X_flat = X_query.reshape(n_queries, -1)
        else:
            n_queries = X_query.shape[0]
            X_flat = X_query.reshape(n_queries, -1)

        distances, indices = self.nn_model.kneighbors(X_flat, n_neighbors=k)
        return distances, indices

    def get_params(self) -> dict:
        """获取模型参数"""
        return {
            'k': self.k,
            'weighted': self.weighted,
            'algorithm': self.algorithm,
            'n_memory': self.memory_X.shape[0] if self.is_fitted else 0,
            'seq_len': self.seq_len if self.is_fitted else 0,
            'pred_len': self.pred_len if self.is_fitted else 0,
            'n_features': self.n_features if self.is_fitted else 0
        }

    def __repr__(self):
        return f"PatternSearch(k={self.k}, weighted={self.weighted}, algorithm='{self.algorithm}')"
