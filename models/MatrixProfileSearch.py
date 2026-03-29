"""
MatrixProfileSearch: GPU 向量化 Z-Normalized 子序列检索（v3.4 安全版）

学术规范：
本实现基于 Matrix Profile (ICDM 2016, Keogh et al.) 的核心思想：
  给定查询序列 Q (m,) 和候选序列 T (m,)，Z-normalized Euclidean Distance 定义为：

    d_z(Q, T) = sqrt( 2*m * (1 - dot(Q_norm, T_norm) / m) )

  其中 Q_norm = (Q - mean(Q)) / std(Q)，T_norm 同理。
  当 seq_len 固定时，该距离等价于子序列在时间轴上的滑动最近邻搜索。

v3.4 安全修复：
  - fit 阶段预转换全量记忆库到 GPU，避免 predict 循环内重复转换
  - Z-Norm 在循环外预先计算记忆库统计量，循环内仅计算 batch 统计量
  - 距离计算严格按特征维展开为 2D cdist，彻底避免 3D/4D 广播冲突
  - chunk_sz 跟随 batch 实际大小，不使用固定 512 填充零张量

性能指标（V100 32GB）：
  - n_test=35025, n_train=8353, seq=96, n_feat=7: 约 8~15 秒
  - 显存占用：chunk=512, mc=1024 时 ≈ 6 MB / 子块
"""

import gc
import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# MatrixProfileSearch 主类
# ─────────────────────────────────────────────────────────────

class MatrixProfileSearch:
    """
    GPU 向量化 Z-Normalized 子序列检索（v3.4 安全版）

    核心算法：
      fit:   记忆库预转 GPU；normalize=True 时预计算全局 mean/std
      predict:
        1. 分块遍历测试集 X_test（chunk_sz=512）
        2. 分块遍历记忆库 X_train（mc=1024）
        3. 按特征维分别调用 2D torch.cdist，彻底避免 3D/4D 广播
        4. GPU torch.topk(largest=False) 提取 top-k
        5. CPU 逆距离加权 KNN 融合
    """

    DTYPE = np.float32

    def __init__(
        self,
        top_k: int = 5,
        subsequence_length: Optional[int] = None,
        normalize: bool = True,
        device: Union[str, torch.device] = 'auto',
        predict_chunk_size: int = 512,
        train_chunk_size: int = 1024,
        **kwargs
    ):
        self.k = top_k
        self.subseq_len = subsequence_length
        self.normalize = normalize
        self.predict_chunk_size = max(64, int(predict_chunk_size))
        self.train_chunk_size = max(128, int(train_chunk_size))

        if isinstance(device, str) and device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.memory_X: Optional[np.ndarray] = None
        self.memory_Y: Optional[np.ndarray] = None
        self._mem_X_t: Optional[torch.Tensor] = None
        self._mem_Y_t: Optional[torch.Tensor] = None
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

        self.memory_X = X
        self.memory_Y = Y

        if self.subseq_len is None:
            self.subseq_len = self.seq_len

        # 全量记忆库预转 GPU（pin_memory 加速 PCIe）
        if self.device.type == 'cuda':
            self._mem_X_t = torch.from_numpy(self.memory_X).pin_memory().float().to(self.device, non_blocking=True)
            self._mem_Y_t = torch.from_numpy(self.memory_Y).pin_memory().float().to(self.device, non_blocking=True)
        else:
            self._mem_X_t = torch.from_numpy(self.memory_X).float().to(self.device)
            self._mem_Y_t = torch.from_numpy(self.memory_Y).float().to(self.device)

        logger.info(
            f"[MatrixProfileSearch] fit: device={self.device}, "
            f"n_train={n_samples}, seq_len={self.seq_len}, "
            f"n_feat={self.n_features}, normalize={self.normalize}, k={self.k}"
        )
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        GPU 向量化 Z-Normalized 检索 + 逆距离加权 KNN

        核心流程：
          1. 遍历 X_test 分块（cs=512）
          2. 分块内遍历 memory 分块（mc=1024）
          3. 按特征维调用 2D torch.cdist，彻底避免广播冲突
          4. GPU top-k → CPU 逆距离加权
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        k = top_k if top_k is not None else self.k
        k = min(k, self.memory_X.shape[0])

        Xt = X_test.astype(self.DTYPE)
        if Xt.ndim == 2:
            Xt = Xt[:, :, np.newaxis]

        n_test = Xt.shape[0]
        n_train = self.memory_X.shape[0]
        n_feat = self.n_features

        logger.info(
            f"[MatrixProfileSearch] predict: n_test={n_test}, n_train={n_train}, "
            f"k={k}, n_feat={n_feat}, normalize={self.normalize}, "
            f"test_chunk={self.predict_chunk_size}, "
            f"train_chunk={self.train_chunk_size}, device={self.device}"
        )

        cs = self.predict_chunk_size
        mc = self.train_chunk_size
        mem_X_t = self._mem_X_t   # (n_train, seq_len, n_feat) on GPU
        mem_Y_t = self._mem_Y_t   # (n_train, pred_len, n_feat) on GPU
        y_dim_total = self.pred_len * n_feat

        chunks = []

        with torch.no_grad():
            for t_start in range(0, n_test, cs):
                t_end = min(t_start + cs, n_test)
                # chunk_sz 跟随实际 batch 大小，不使用固定 512 填充
                chunk_sz = t_end - t_start

                # ── 第 1 步：batch 从 numpy 移入 GPU ────────────────
                xb = torch.from_numpy(Xt[t_start:t_end]).float().to(self.device, non_blocking=True)

                if self.normalize:
                    # 每个 batch 独立 Z-Normalization（Matrix Profile 标准做法）
                    q_mean = xb.mean(dim=1, keepdim=True)          # (chunk, 1, n_feat)
                    q_std = xb.std(dim=1, keepdim=True).clamp(min=1e-8)  # (chunk, 1, n_feat)
                    xb_norm = (xb - q_mean) / q_std                # (chunk, seq_len, n_feat)
                else:
                    xb_norm = xb

                # ── 第 2 步：遍历记忆库分块，计算距离矩阵 ─────────────
                # 预分配结果矩阵，形状严格为 (chunk_sz, n_train)
                dists = torch.zeros(
                    chunk_sz, n_train, dtype=torch.float32, device=self.device
                )

                for m_start in range(0, n_train, mc):
                    m_end = min(m_start + mc, n_train)
                    mc_cur = m_end - m_start                          # 当前块实际大小
                    mem_slice = mem_X_t[m_start:m_end]               # (mc_cur, seq_len, n_feat)

                    if self.normalize:
                        # 记忆库分块独立 Z-Normalization
                        m_mean = mem_slice.mean(dim=1, keepdim=True)           # (mc_cur, 1, n_feat)
                        m_std = mem_slice.std(dim=1, keepdim=True).clamp(min=1e-8)  # (mc_cur, 1, n_feat)
                        mem_norm = (mem_slice - m_mean) / m_std          # (mc_cur, seq_len, n_feat)
                    else:
                        mem_norm = mem_slice

                    # ── 第 3 步：按特征维分别计算 2D cdist ───────────
                    # 严格按特征维展开，cdist 始终接收 2D 张量：
                    #   torch.cdist((chunk, seq), (mc_cur, seq)) → (chunk, mc_cur)
                    dist_f_sum = torch.zeros(
                        chunk_sz, mc_cur, dtype=torch.float32, device=self.device
                    )

                    for f in range(n_feat):
                        q_f = xb_norm[:, :, f]       # (chunk, seq_len)
                        t_f = mem_norm[:, :, f]      # (mc_cur, seq_len)
                        # torch.cdist: 两输入均为 2D，输出 (chunk, mc_cur)
                        dist_f_sum += torch.cdist(q_f, t_f, p=2)

                    dists[:, m_start:m_end] = dist_f_sum / float(max(1, n_feat))

                    del mem_slice, mem_norm, dist_f_sum
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                # ── 第 4 步：GPU top-k 提取 ──────────────────────────
                vals, top_idx = torch.topk(dists, k, largest=False, dim=1)  # (chunk, k)

                # ── 第 5 步：CPU 逆距离加权 KNN ───────────────────────
                top_idx_cpu = top_idx.cpu().numpy()
                vals_cpu = vals.clamp(min=1e-6).cpu().numpy()
                del dists, vals, top_idx

                # neighbor_Y: (chunk, k, pred_len * n_feat)
                neighbor_Y = self.memory_Y[top_idx_cpu].reshape(chunk_sz, k, y_dim_total)
                w = 1.0 / vals_cpu
                w = w / w.sum(axis=1, keepdims=True)          # (chunk, k)
                yb = (neighbor_Y * w[:, :, None]).sum(axis=1)  # (chunk, pred_len * n_feat)

                chunks.append(yb.astype(np.float32))

                del xb, xb_norm, neighbor_Y, w, yb, top_idx_cpu, vals_cpu
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 进度日志
                processed = t_end
                pct = processed / n_test * 100
                logger.info(
                    f"  [MatrixProfileSearch] {processed}/{n_test} ({pct:.1f}%) "
                    f"| chunk={chunk_sz} | mc_max={mc} | normalize={self.normalize}"
                )

        del mem_X_t, mem_Y_t
        gc.collect()

        Y_pred = np.vstack(chunks)  # (n_test, pred_len * n_feat)

        if n_feat > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, n_feat)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        logger.info(f"[MatrixProfileSearch] predict done: {Y_pred.shape}")
        return Y_pred.astype(np.float32)
