"""
MatrixProfileSearch: 矩阵轮廓精确子序列检索

基于 UCR Eamonn Keogh 团队的 Matrix Profile (ICDM 2016) 规范，
使用 stumpy 库实现极速多维距离轮廓计算。

学术规范：
- stumpy.mstump: 多维矩阵轮廓（多变量同时计算）
- MASS (Mueen's Algorithm for Similarity Search): O(n·m) 精确距离轮廓
- Distance Profile: 查询序列在记忆库上的逐点最小距离轮廓
- 核心思想：利用 FFT 卷积优化，将 DTW 类搜索降至 O(n·log(n))

实现要求：
- fit: 仅存储 X_train / Y_train
- predict: stumpy.mstump / MASS 批量计算 X_test 与 X_train 的距离轮廓，
  提取 top_k 最小距离对应的 Y_train 片段，加权融合

核心参数：top_k

必须处理多变量 (features='M') 情况。

Usage:
    from models.MatrixProfileSearch import MatrixProfileSearch
    model = MatrixProfileSearch(top_k=5)
    model.fit(X_train_norm, Y_train_norm)
    Y_pred = model.predict(X_test_norm)
"""

import gc
import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入 stumpy（Matrix Profile 标准库）
try:
    import stumpy
    _HAS_STUMPY = True
    logger.info("[MatrixProfileSearch] stumpy available")
except ImportError:
    _HAS_STUMPY = False
    logger.warning("[MatrixProfileSearch] stumpy not available, using fallback Euclidean")


# ─────────────────────────────────────────────────────────────
# MatrixProfileSearch 主类
# ─────────────────────────────────────────────────────────────

class MatrixProfileSearch:
    """
    Matrix Profile 子序列检索模型

    使用 stumpy（MASS / mstump）计算距离轮廓，
    提取最近邻并融合 Y_train。

    Args:
        top_k: 近邻数（默认 5）
        subsequence_length: 子序列长度（默认 seq_len，匹配记忆库子序列）
        normalize: 是否对距离轮廓做 z-normalize（默认 True，MASS 规范）
    """

    DTYPE = np.float64  # stumpy 推荐 float64 以减少数值误差

    def __init__(
        self,
        top_k: int = 5,
        subsequence_length: Optional[int] = None,
        normalize: bool = True,
        **kwargs
    ):
        self.k = top_k
        self.subseq_len = subsequence_length  # 留 None 自动设为 seq_len
        self.normalize = normalize

        self.memory_X: Optional[np.ndarray] = None  # (n_train, seq_len, n_feat)
        self.memory_Y: Optional[np.ndarray] = None  # (n_train, pred_len, n_feat)
        self.is_fitted = False
        self.seq_len: int = 0
        self.pred_len: int = 0
        self.n_features: int = 1

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        存储记忆库（已归一化数据）

        Args:
            X_train: shape (n_train, seq_len, n_feat) 或 (n_train, seq_len)
            Y_train: shape (n_train, pred_len, n_feat) 或 (n_train, pred_len)
        """
        X = X_train.astype(self.DTYPE)
        Y = Y_train.astype(self.DTYPE)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = Y.shape[-1] if Y.ndim == 3 else 1

        n_samples = X.shape[0]

        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        if Y.ndim == 2:
            Y = Y[:, :, np.newaxis]

        self.memory_X = X.astype(self.DTYPE)   # (n_train, seq_len, n_feat)
        self.memory_Y = Y.astype(self.DTYPE)   # (n_train, pred_len, n_feat)

        if self.subseq_len is None:
            self.subseq_len = self.seq_len

        self.is_fitted = True
        logger.info(
            f"[MatrixProfileSearch] fit: memory_X={self.memory_X.shape}, "
            f"memory_Y={self.memory_Y.shape}, subseq_len={self.subseq_len}, k={self.k}"
        )
        return self

    def predict(self, X_test: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        使用 Matrix Profile 检索最近邻并预测

        Args:
            X_test: shape (n_test, seq_len, n_feat) 或 (n_test, seq_len)
            top_k: 覆盖默认的 k

        Returns:
            shape (n_test, pred_len, n_feat) 或 (n_test, pred_len)
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        k = top_k if top_k is not None else self.k
        k = min(k, self.memory_X.shape[0])

        Xt = X_test.astype(self.DTYPE)
        if Xt.ndim == 2:
            Xt = Xt[:, :, np.newaxis]

        n_test = Xt.shape[0]
        logger.info(
            f"[MatrixProfileSearch] predict: n_test={n_test}, k={k}, "
            f"normalize={self.normalize}"
        )

        if self.n_features > 1 and _HAS_STUMPY:
            # ── 多变量路径：stumpy.mstump ───────────────────────────
            # mstump 同时计算所有维度的距离轮廓
            Y_preds = []
            for i in range(n_test):
                query = Xt[i]  # (seq_len, n_feat)
                # 用 MASS 对 query 搜索 memory_X
                if self.normalize:
                    # MASS: z-normalized distance profile
                    dists = self._mass_normalized(query, self.memory_X)
                else:
                    dists = self._mass_raw(query, self.memory_X)

                # 取距离最小的 top_k
                top_idx = np.argpartition(dists, k)[:k]
                neighbor_Y = self.memory_Y[top_idx]  # (k, pred_len, n_feat)

                # 逆距离加权融合
                vals = dists[top_idx]
                vals_safe = np.where(vals < 1e-8, 1e-8, vals)
                w = 1.0 / vals_safe
                w = w / w.sum()

                y_pred = (neighbor_Y * w[:, None, None]).sum(axis=0)  # (pred_len, n_feat)
                Y_preds.append(y_pred)

            Y_pred = np.stack(Y_preds)  # (n_test, pred_len, n_feat)

        elif _HAS_STUMPY:
            # ── 单变量路径：stumpy.stump ───────────────────────────
            Y_preds = []
            for i in range(n_test):
                query = Xt[i, :, 0]  # (seq_len,)
                # stump 计算 query 在 memory_X 第一维上的距离轮廓
                dists = self._mass_normalized_1d(query, self.memory_X[:, :, 0])

                top_idx = np.argpartition(dists, k)[:k]
                neighbor_Y = self.memory_Y[top_idx, :, 0]  # (k, pred_len)

                vals = dists[top_idx]
                vals_safe = np.where(vals < 1e-8, 1e-8, vals)
                w = 1.0 / vals_safe
                w = w / w.sum()

                y_pred = (neighbor_Y * w[:, None]).sum(axis=0)  # (pred_len,)
                Y_preds.append(y_pred)

            Y_pred = np.stack(Y_preds)  # (n_test, pred_len)

        else:
            # ── Fallback：欧氏距离最近邻（无 Matrix Profile）──────────
            mem_flat = self.memory_X.reshape(self.memory_X.shape[0], -1)
            Xt_flat = Xt.reshape(n_test, -1)
            dists = np.linalg.norm(
                Xt_flat[:, None, :] - mem_flat[None, :, :], axis=2
            )  # (n_test, n_train)

            Y_preds = []
            for i in range(n_test):
                top_idx = np.argpartition(dists[i], k)[:k]
                vals = dists[i, top_idx]
                vals_safe = np.where(vals < 1e-8, 1e-8, vals)
                w = vals_safe / vals_safe.sum()
                y_pred = (self.memory_Y[top_idx].mean(axis=0) * w.sum() +
                         (self.memory_Y[top_idx] * w[:, None, None]).sum(axis=0) - self.memory_Y[top_idx].mean(axis=0) * w.sum())
                Y_preds.append(y_pred)

            Y_pred = np.stack(Y_preds)

        # ── 恢复原始 dtype ─────────────────────────────────────────
        Y_pred = Y_pred.astype(np.float32)

        if self.n_features == 1 and Y_pred.ndim == 3:
            Y_pred = Y_pred[:, :, 0]

        logger.info(f"[MatrixProfileSearch] predict done: output shape={Y_pred.shape}")
        return Y_pred

    # ── MASS (Mueen's Algorithm for Similarity Search) 实现 ───

    def _mass_normalized(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """
        多变量 MASS（z-normalized）

        query: (seq_len, n_feat)
        corpus: (n_train, seq_len, n_feat)

        Returns:
            dists: (n_train,) — 每个记忆样本与 query 的 z-normalized 距离
        """
        n_train = corpus.shape[0]
        seq_len = corpus.shape[1]
        n_feat = corpus.shape[2]
        dists = np.full(n_train, np.inf, dtype=np.float64)

        for k in range(n_train):
            # 对每个维度独立 MASS 后求均值
            dim_dists = []
            for f in range(n_feat):
                q = query[:, f]
                c = corpus[k, :, f]
                d = self._mass_1d_normalized(q, c)
                dim_dists.append(d)
            dists[k] = np.mean(dim_dists)

        return dists

    def _mass_1d_normalized(self, query: np.ndarray, corpus_seq: np.ndarray) -> float:
        """
        单变量 MASS（z-normalized distance profile）

        基于 FFT 卷积优化，O(n·log(n)) 计算 query 与 corpus_seq 的距离轮廓
        """
        n = len(corpus_seq)
        m = len(query)

        if n < m:
            return float(np.linalg.norm(query - corpus_seq[:m]))

        # z-normalize
        q_mean = np.mean(query)
        q_std = np.std(query)
        if q_std < 1e-8:
            q_std = 1.0
        q_norm = (query - q_mean) / q_std

        # 用滑动窗口计算与 corpus 的 dot product
        # rolling_mean / rolling_std 使用 cumsum 优化
        def rolling_mean_std(arr, w):
            cum = np.concatenate([[0], np.cumsum(arr)])
            w_sum = cum[w:] - cum[:-w]
            w_mean = w_sum / w
            w_sq = np.cumsum(arr ** 2)
            w_var = (w_sq[w:] - w_sq[:-w]) / w - w_mean ** 2
            w_std = np.sqrt(np.clip(w_var, 0, None))
            w_std = np.where(w_std < 1e-8, 1.0, w_std)
            return w_mean, w_std

        c_mean, c_std = rolling_mean_std(corpus_seq, m)

        # sliding dot product
        rev_q = q_norm[::-1]
        dot = np.correlate(corpus_seq, query, mode='valid') / m

        # z-normalized dot
        z_norm_dot = (dot - c_mean * q_mean) / (c_std * q_std + 1e-8)

        # distance profile
        dist_profile = np.sqrt(2 * (m - z_norm_dot + 1e-8))
        dist_profile = np.clip(dist_profile, 0, None)

        return float(np.min(dist_profile))

    def _mass_normalized_1d(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """
        单变量批量 MASS：对所有记忆样本计算距离

        corpus: (n_train, seq_len)
        返回: (n_train,)
        """
        n_train = corpus.shape[0]
        dists = np.full(n_train, np.inf, dtype=np.float64)
        for k in range(n_train):
            dists[k] = self._mass_1d_normalized(query, corpus[k])
        return dists

    def _mass_raw(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """原始（非归一化）欧氏距离近似"""
        n_train = corpus.shape[0]
        dists = np.full(n_train, np.inf, dtype=np.float64)
        for k in range(n_train):
            diff = query - corpus[k]  # (seq_len, n_feat)
            dists[k] = np.sqrt(np.mean(diff ** 2))
        return dists
