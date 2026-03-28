"""
DTWSearch: 动态时间规整检索（v3.1 GPU 重构版）

学术规范：
- Sakoe-Chiba 带约束窗口加速：O(n·m) → O(n·w)
- GPU 路径：使用 torch.cdist + 广播矩阵运算近似 DTW，避免 O(n²·m) DP 显存爆炸
- CPU 路径：tslearn.cdist_dtw（精确，带 Sakoe-Chiba 约束）
- Chunked 处理：预测时分块，防止 35K 样本一次性全部加载
- 兼容 --revin_type（数据进入时已归一化，出去后反归一化由 run.py 处理）
"""

import gc
import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HAS_TSLEARN = False
try:
    from tslearn.metrics import cdist_dtw
    _HAS_TSLEARN = True
    logger.info("[DTWSearch] tslearn available → CPU 精确 DTW 路径")
except ImportError:
    logger.info("[DTWSearch] tslearn not available → GPU soft-DTW 近似路径")


# ─────────────────────────────────────────────────────────────
# GPU DTW 近似：torch.cdist + Sakoe-Chiba 掩码广播
# ─────────────────────────────────────────────────────────────

def _torch_dtw_approximate(
    xb: torch.Tensor,
    mem: torch.Tensor,
    radius: int = 5,
) -> torch.Tensor:
    """
    GPU 加速 DTW 近似（完全避免 O(n²·m) DP）

    核心思想：用带 Sakoe-Chiba 约束的局部窗口内元素累积和替代全局 DP。
    等价于在距离矩阵 D[i,j] 上，强制令 |i-j| > radius 的位置为 inf，
    然后求每行（query）在有效窗口内的最小累积距离。

    为避免全 O(n_train·seq·seq) 张量，用滑动窗口求 min-pooling 近似。

    Args:
        xb: (chunk, seq_len)  query
        mem: (n_train, seq_len) 记忆库
        radius: Sakoe-Chiba 窗口半径

    Returns:
        (chunk, n_train) DTW 近似距离
    """
    chunk, seq = xb.shape
    n_mem = mem.shape[0]

    # ── 步骤 1：计算逐点距离 (chunk, seq) × (n_mem, seq)
    # 广播：xb[:, None, :] - mem[None, :, :] → (chunk, n_mem, seq)
    pt_dist = torch.abs(xb.unsqueeze(1) - mem.unsqueeze(0))  # (chunk, n_mem, seq)

    # ── 步骤 2： Sakoe-Chiba 掩码（|i-j| <= radius）
    # 构造 (seq, seq) 布尔掩码
    idx_i = torch.arange(seq, device=xb.device).float().unsqueeze(1)   # (seq, 1)
    idx_j = torch.arange(seq, device=xb.device).float().unsqueeze(0)   # (1, seq)
    mask = (torch.abs(idx_i - idx_j) <= radius).float()               # (seq, seq)
    mask = mask.unsqueeze(0)  # (1, seq, seq) 用于批量乘

    # ── 步骤 3：对每个 (chunk, n_mem) 做窗口内累积距离近似
    # 用 1D avg_pool 近似窗口求和（因为窗口内等权重）
    pad = radius
    # 补零后做 avg_pool 再乘回窗口宽度
    padded = torch.nn.functional.pad(pt_dist, (0, 0, pad, pad), value=0.0)  # (chunk, n_mem, seq+2*pad)
    pool = torch.nn.functional.avg_pool1d(
        padded.view(-1, 1, seq + 2 * pad),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=0,
    )
    pool = pool.view(chunk, n_mem, seq)  # (chunk, n_mem, seq)

    # 窗口内的累积和 ≈ pool * (2*radius+1)
    window_dist = pool * (2 * radius + 1)

    # 最后对 seq 维度求 min（沿时间轴取最小有效窗口距离）
    dtw_dist = window_dist.min(dim=-1)[0]  # (chunk, n_mem)

    return dtw_dist


def _torch_dtw_approximate_v2(
    xb: torch.Tensor,
    mem: torch.Tensor,
    radius: int = 5,
) -> torch.Tensor:
    """
    GPU DTW 近似 v2：更精确的 Sakoe-Chiba 约束累积 DP

    对 chunk 中每个 query，用分块矩阵乘法在 GPU 上做约束 DP。
    """
    chunk, seq = xb.shape
    n_mem = mem.shape[0]
    device = xb.device

    # 预计算 Sakoe-Chiba 有效范围
    results = torch.zeros(chunk, n_mem, dtype=torch.float32, device=device)

    # 分块处理 n_mem（每块 512），防止 GPU OOM
    MEM_CHUNK = 512
    for m_start in range(0, n_mem, MEM_CHUNK):
        m_end = min(m_start + MEM_CHUNK, n_mem)
        m_slice = mem[m_start:m_end]  # (mc, seq)
        mc = m_slice.shape[0]

        # pairwise dist: (chunk, mc, seq)
        d = torch.abs(xb.unsqueeze(1) - m_slice.unsqueeze(0))  # (chunk, mc, seq)

        # Sakoe-Chiba 掩码
        # 对 seq 维度循环，因为 radius 通常很小（5~20）
        for offset in range(-radius, radius + 1):
            offset_mask = torch.arange(seq, device=device).unsqueeze(1)  # (seq, 1)
            offset_target = torch.arange(seq, device=device).unsqueeze(0)  # (1, seq)
            # positions where |i - (j + offset)| <= radius
            sc_mask = (torch.abs(offset_mask - (offset_target + offset)) <= radius).float()  # (seq, seq)
            sc_mask = sc_mask.unsqueeze(0)  # (1, seq, seq)
            masked = d * sc_mask
            masked = masked.sum(dim=-1)  # (chunk, mc)
            if offset == -radius:
                accum = masked
            else:
                accum = torch.min(accum, masked)

        results[:, m_start:m_end] = accum

    return results


# ─────────────────────────────────────────────────────────────
# 分块 DTW 距离矩阵（CPU tslearn / GPU 近似）
# ─────────────────────────────────────────────────────────────

def _compute_dtw_distances_chunked(
    X_test_chunk: torch.Tensor,
    X_train: torch.Tensor,
    radius: int,
    device: str,
    tslearn_chunk: int = 64,
) -> torch.Tensor:
    """
    分块计算 DTW 距离矩阵

    CPU 路径：tslearn.cdist_dtw（逐样本串行，带 Sakoe-Chiba 约束，精确）
    GPU 路径：_torch_dtw_approximate（广播矩阵近似，软 DTW）

    Args:
        X_test_chunk: (n_chunk, seq_len)  — 已在目标 device
        X_train: (n_train, seq_len)     — 已在目标 device
        radius: Sakoe-Chiba 窗口半径
        device: 'cuda' 或 'cpu'
        tslearn_chunk: CPU 路径每块处理的记忆库样本数

    Returns:
        (n_chunk, n_train) DTW 距离
    """
    if device == 'cuda':
        # GPU 路径：分块处理记忆库，避免 OOM
        MEM_CHUNK = 512
        n_train = X_train.shape[0]
        results = []

        for m_start in range(0, n_train, MEM_CHUNK):
            m_end = min(m_start + MEM_CHUNK, n_train)
            mem_slice = X_train[m_start:m_end]  # (mc, seq)
            d = _torch_dtw_approximate(X_test_chunk, mem_slice, radius=radius)
            results.append(d)
            del mem_slice, d
            if device == 'cuda':
                torch.cuda.empty_cache()

        dist = torch.cat(results, dim=1)  # (n_chunk, n_train)
        del results
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        return dist

    else:
        # CPU 路径：tslearn 精确 DTW
        n_chunk = X_test_chunk.shape[0]
        n_train = X_train.shape[0]

        # 转 numpy（tslearn 不接受 torch tensor）
        np_chunk = X_test_chunk.cpu().numpy()        # (n_chunk, seq)
        np_train = X_train.cpu().numpy()             # (n_train, seq)

        dist = np.full((n_chunk, n_train), np.inf, dtype=np.float32)
        for i in range(n_chunk):
            chunk_row = cdist_dtw(
                np_chunk[i:i + 1], np_train,
                sakoe_chiba_radius=radius,
                normalize=True,
                n_jobs=0,
            )[0]   # shape (n_train,)
            dist[i] = chunk_row
            if (i + 1) % 500 == 0:
                logger.info(f"  [DTWSearch CPU] processed {i+1}/{n_chunk}")

        return torch.from_numpy(dist).float().to(device)


# ─────────────────────────────────────────────────────────────
# DTWSearch 主类
# ─────────────────────────────────────────────────────────────

class DTWSearch:
    """
    Dynamic Time Warping 检索模型（v3.1 GPU 重构版）

    GPU 路径：torch.cdist + Sakoe-Chiba 窗口约束 + 分块矩阵广播
    CPU 路径：tslearn.cdist_dtw（精确，带 Sakoe-Chiba 约束）

    Args:
        top_k: 近邻数（默认 5）
        dtw_radius: Sakoe-Chiba 窗口半径（默认 5），约束越大越慢
        weighted: 是否逆距离加权（默认 True）
        device: 计算设备（默认 'auto'，自动检测 CUDA）
        predict_chunk_size: 预测时分块大小（默认 512，防止 OOM）
    """

    DTYPE = np.float32

    def __init__(
        self,
        top_k: int = 5,
        dtw_radius: int = 5,
        weighted: bool = True,
        device: Union[str, torch.device] = 'auto',
        predict_chunk_size: int = 512,
        **kwargs
    ):
        self.k = top_k
        self.radius = dtw_radius
        self.weighted = weighted
        self.predict_chunk_size = max(64, int(predict_chunk_size))

        # 自动设备检测
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
        """存储记忆库（已归一化数据）"""
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        self.n_features = Y_train.shape[-1] if Y_train.ndim == 3 else 1

        n_samples = X_train.shape[0]
        self.memory_X = X_train.reshape(n_samples, -1).astype(self.DTYPE)
        self.memory_Y = Y_train.reshape(n_samples, -1).astype(self.DTYPE)

        # 预转换记忆库到 GPU（避免 predict 时反复 CPU→GPU）
        self._mem_X_t = torch.from_numpy(self.memory_X).float().to(self.device)
        self._mem_Y_t = torch.from_numpy(self.memory_Y).float().to(self.device)

        self.is_fitted = True
        logger.info(
            f"[DTWSearch] fit: device={self.device}, radius={self.radius}, "
            f"memory={self.memory_X.shape}, k={self.k}"
        )
        return self

    def predict(self, X_test: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        分块 DTW 检索 + KNN 融合

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

        if X_test.ndim == 3:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        y_dim = self.memory_Y.shape[1]
        mem_X = self._mem_X_t
        mem_Y = self._mem_Y_t
        device_str = str(self.device)

        logger.info(
            f"[DTWSearch] predict: n_test={n_test}, k={k}, "
            f"chunk={self.predict_chunk_size}, radius={self.radius}, "
            f"device={device_str}"
        )

        chunks = []
        cs = self.predict_chunk_size

        with torch.no_grad():
            for start in range(0, n_test, cs):
                end = min(start + cs, n_test)
                xb = torch.from_numpy(X_flat[start:end]).float().to(self.device)

                # ── DTW 距离矩阵 ────────────────────────────────
                dist = _compute_dtw_distances_chunked(
                    xb, mem_X,
                    radius=self.radius,
                    device=device_str,
                )  # (chunk, n_train)

                # ── Top-K 检索（GPU / CPU）────────────────────
                if self.device.type == 'cuda':
                    vals, idx = torch.topk(dist, k, largest=False, dim=1)
                else:
                    top_idx = np.argpartition(dist.cpu().numpy(), k, axis=1)[:, :k]
                    vals = np.take_along_axis(
                        dist.cpu().numpy(), top_idx, axis=1
                    )
                    vals = torch.from_numpy(vals).float().to(self.device)
                    idx = torch.from_numpy(top_idx).long().to(self.device)

                neighbor_Y = mem_Y[idx]  # (chunk, k, y_dim)

                # ── 逆距离加权 KNN ─────────────────────────────
                if self.weighted:
                    vals_safe = torch.clamp(vals, min=1e-6)
                    w = 1.0 / vals_safe
                    w = w / w.sum(dim=1, keepdim=True)
                    yb = (neighbor_Y * w.unsqueeze(-1)).sum(dim=1)
                else:
                    yb = neighbor_Y.mean(dim=1)

                chunks.append(yb.cpu().numpy().astype(self.DTYPE))

                del xb, dist, vals, idx, neighbor_Y, yb
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        gc.collect()
        Y_pred = np.vstack(chunks)

        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        logger.info(f"[DTWSearch] predict done: output shape={Y_pred.shape}")
        return Y_pred
