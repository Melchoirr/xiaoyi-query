"""
PatternSearch: 基于记忆检索的时序预测基线模型
使用 k-NN / KD-Tree 近邻检索算法寻找相似波形并进行预测

Bug 修复 (v2.1):
- __init__ 显式接收 top_k, weighted 参数（兼容 run.py 的命名约定）
- 所有 __init__ 参数末尾追加 **kwargs，安全提取未声明参数，防止 TypeError
"""

import numpy as np
import gc
import logging
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class PatternSearch:
    """
    基于记忆检索的时序预测基线模型

    核心思想：
    - 将历史输入序列 X 作为记忆库
    - 对每个测试样本，通过 k-NN 找到最相似的 K 个历史序列
    - 使用对应历史序列的真实输出 Y 进行加权融合预测
    """

    DTYPE = np.float32   # 全局统一 float32

    def __init__(
        self,
        # ── 兼容 run.py 传入的 top_k ──
        top_k: int = 5,
        weighted: bool = True,
        algorithm: str = 'kd_tree',
        # ── 基础维度参数（预留，暂未使用）──
        seq_len: int = 0,
        pred_len: int = 0,
        n_features: int = 1,
        # ── 安全吸收未声明参数，防止 TypeError ──
        **kwargs
    ):
        """
        Args:
            top_k: 近邻数量（兼容 run.py 的参数命名）
            weighted: 是否使用逆距离加权，True为逆距离加权，False为简单平均
            algorithm: 近邻搜索算法，'kd_tree', 'ball_tree', 'brute', 'auto'
            seq_len: 输入序列长度（预留）
            pred_len: 预测序列长度（预留）
            n_features: 特征数量（预留）
            **kwargs: 安全吸收未声明参数
        """
        self.k = top_k      # 内部用 k，兼容传入的 top_k
        self.weighted = weighted
        self.algorithm = algorithm
        self.nn_model = None
        self.memory_X = None
        self.memory_Y = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        构建记忆库，使用训练集的 (X, Y) 样本

        内存优化：
        - 全部使用 float32
        - 不再需要复制数据（直接 astype）
        """
        logger.info(f"[PatternSearch] fit: X={X_train.shape}, Y={Y_train.shape}")
        n_samples = X_train.shape[0]

        # 展平 + 强制 float32
        if X_train.ndim == 3:
            self.seq_len = X_train.shape[1]
            self.n_features = X_train.shape[2]
            self.memory_X = X_train.reshape(n_samples, -1).astype(self.DTYPE)
        else:
            self.seq_len = X_train.shape[1]
            self.n_features = 1
            self.memory_X = X_train.reshape(n_samples, -1).astype(self.DTYPE)

        if Y_train.ndim == 3:
            self.pred_len = Y_train.shape[1]
            self.memory_Y = Y_train.reshape(n_samples, -1).astype(self.DTYPE)
        else:
            self.pred_len = Y_train.shape[1]
            self.n_features = 1
            self.memory_Y = Y_train.reshape(n_samples, -1).astype(self.DTYPE)

        # 使用 sklearn NearestNeighbors 构建 KD-Tree 索引
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.k, n_samples),
            algorithm=self.algorithm,
            metric='euclidean',
            n_jobs=-1
        )
        self.nn_model.fit(self.memory_X)
        self.is_fitted = True

        logger.info(f"[PatternSearch] fitted: memory_X={self.memory_X.shape}, "
                    f"memory_Y={self.memory_Y.shape}")

        return self

    def predict(self, X_test: np.ndarray, top_k: int = None) -> np.ndarray:
        """
        对测试样本进行预测

        内存优化：所有中间变量使用 float32，结果保持 float32
        返回 shape: (n_test, pred_len) 或 (n_test, pred_len, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        k = top_k if top_k is not None else self.k
        k = min(k, self.memory_X.shape[0])

        # 展平测试输入
        if X_test.ndim == 3:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        logger.info(f"[PatternSearch] predict: {n_test} samples, k={k}")

        # 在记忆库中搜索 k 个最近邻
        distances, indices = self.nn_model.kneighbors(X_flat, n_neighbors=k)

        # 检索对应的 Y 值（memory_Y 已是 float32）
        neighbor_Y = self.memory_Y[indices]  # shape: [n_test, k, y_dim]

        # 融合策略
        if self.weighted:
            # 逆距离加权平均
            distances = np.clip(distances, 1e-10, None)
            weights = 1.0 / distances
            weights = weights / weights.sum(axis=1, keepdims=True)
            Y_pred = np.sum(neighbor_Y * weights[:, :, np.newaxis], axis=1)
        else:
            Y_pred = np.mean(neighbor_Y, axis=1)

        # 释放测试展平数组
        del X_flat, neighbor_Y, distances
        gc.collect()

        # 恢复原始形状（与 fit 时 n_features 对齐）
        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        logger.info(f"[PatternSearch] predict done: {Y_pred.shape}")
        return Y_pred

    def get_neighbors(self, X_query: np.ndarray, top_k: int = None):
        """获取查询样本的 k 个最近邻及其距离"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        k = top_k if top_k is not None else self.k
        k = min(k, self.memory_X.shape[0])

        if X_query.ndim == 3:
            n_queries = X_query.shape[0]
            X_flat = X_query.reshape(n_queries, -1).astype(self.DTYPE)
        else:
            n_queries = X_query.shape[0]
            X_flat = X_query.reshape(n_queries, -1).astype(self.DTYPE)

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
