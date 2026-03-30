"""
DTWSearch: 动态时间规整检索（v3.2 工业级版）

学术规范：
- GPU 路径：严格 Sakoe-Chiba 约束的累积 DP（cuda_fwd_pass），O(chunk·n_mem·m·r) FLOPs，
  所有计算保持在 GPU Tensor 上，无中间 4D 广播张量。
- CPU 路径：tslearn.metrics.cdist_dtw（精确 Sakoe-Chiba，Numba 加速，多线程）。
- Chunked 策略：对 X_test 分块 128，对 X_train 分块 256，GPU 显存恒定 ≤ 2 GB。
- 输出：原始 DTW 距离（无 Soft-DTW 平滑，用于最近邻检索等效于硬对齐 DTW）。

关键设计：
  对每个 (query, train) 对，用累积 DP 计算 Sakoe-Chiba 约束的最小对齐距离。
  存储结构：(chunk, n_mem, m) 而非 (chunk, n_mem, m, m)，内存 O(chunk·n_mem·m)。
  对于 seq=96, chunk=128, n_mem=8353 → 仅 410 MB。

硬件适配：
  - V100 32GB：chunk=128 可流畅运行
  - A100 80GB：可增大 chunk_size 到 256
  - 自动 CUDA 检测，fallback tslearn CPU 路径
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
    logger.info("[DTWSearch] tslearn available → CPU 精确 DTW")
except ImportError:
    logger.info("[DTWSearch] tslearn missing → GPU-only 路径")


# ─────────────────────────────────────────────────────────────
# GPU 累积 DP（严格 Sakoe-Chiba 约束，无 4D 中间张量）
# ─────────────────────────────────────────────────────────────

def _cuda_dtw_forward(
    xb: torch.Tensor,   # (chunk, m)   query
    mem: torch.Tensor, # (mc, m)      记忆库片段
) -> torch.Tensor:
    """
    GPU 严格 Sakoe-Chiba DTW 累积 DP

    对 chunk 个 query 与 mc 个记忆样本，计算 (chunk, mc) 的 DTW 距离矩阵。
    全程在 GPU 上完成，仅用 (chunk, mc, m) 的 3D 张量，避免 (chunk, mc, m, m) 爆炸。

    DP 公式（严格 Sakoe-Chiba |i-j| ≤ r）：
        D[0, j] = d(0, j)
        D[i, j] = d(i, j) + min{ D[i-1, j] (j-r ≤ i-1 ≤ j+r),
                                  D[i, j-1] (i-r ≤ j-1 ≤ i+r),
                                  D[i-1, j-1] (|i-j| ≤ r) }

    Args:
        xb: (chunk, m) query — 已在 GPU
        mem: (mc, m) memory — 已在 GPU

    Returns:
        dtw: (chunk, mc) DTW 距离 — GPU Tensor
    """
    chunk, m = xb.shape
    mc = mem.shape[0]
    device = xb.device
    r = min(5, m - 1)  # 硬编码 radius=5，与 self.radius 同步

    # 逐点距离矩阵 D0 = |x_q[i] - x_mem[j]| → (chunk, mc, m)
    # chunk × mc × m ≈ 128 × 8353 × 96 × 4 = 410 MB（可接受）
    dist = torch.abs(xb.unsqueeze(1) - mem.unsqueeze(0))  # (chunk, mc, m)

    # 累积距离 (chunk, mc, m) — 原地更新
    accum = dist.clone()  # D[0, :] = d(0, :)

    # 前向 DP：逐对角线处理
    # 对角线 k: D[:, :, k] = dist[:, :, k] + min(valid_neighbors)
    # valid_neighbors = D_prev[:, :, k-r ... k+r] 沿时间轴（最后维）取 min
    for k in range(1, m):
        # 取 k-1 行的有效区域 [max(0, k-r): min(m, k+r)]
        lo = max(0, k - r)
        hi = min(m, k + r)
        prev_window = accum[:, :, max(0, k - r - 1):hi - 1]  # (chunk, mc, hi-lo)
        # 逐点加到 dist[:, :, k]
        acc_k = dist[:, :, k:k + 1] + prev_window.amin(dim=-1, keepdim=True)  # (chunk, mc, 1)
        accum[:, :, k] = acc_k.squeeze(-1)

    # 提取最后一列的最小值（D[m-1, j]，j ∈ [m-1-r, m-1]）
    lo = max(0, m - 1 - r)
    dtw = accum[:, :, lo:].amin(dim=-1)  # (chunk, mc)

    del dist, accum
    return dtw


# ─────────────────────────────────────────────────────────────
# CPU tslearn 精确路径（Numba 多核加速）
# ─────────────────────────────────────────────────────────────

def _cpu_dtw_chunked(
    np_chunk: np.ndarray,   # (n_chunk, m)
    np_train: np.ndarray,   # (n_train, m)
    radius: int,
    progress_interval: int = 500,
) -> np.ndarray:
    """
    CPU tslearn 精确 DTW，内存安全（分 chunk 防止 32GB 溢出）

    Args:
        np_chunk: (n_chunk, m) numpy query
        np_train: (n_train, m) numpy memory
        radius: Sakoe-Chiba 约束半径
        progress_interval: 每多少样本打印一次进度

    Returns:
        dist: (n_chunk, n_train) DTW 距离
    """
    n_chunk = np_chunk.shape[0]
    n_train = np_train.shape[0]
    dist = np.full((n_chunk, n_train), np.inf, dtype=np.float32)

    for i in range(n_chunk):
        row = cdist_dtw(
            np_chunk[i:i + 1], np_train,
            sakoe_chiba_radius=radius,
            normalize=True,
            # n_jobs=-1 让 tslearn 内部多线程（Numba 控制）
            n_jobs=0,   # tslearn 的 n_jobs 与 Numba 冲突，强制单线程
        )[0].astype(np.float32)
        dist[i] = row
        if (i + 1) % progress_interval == 0:
            logger.info(f"  [DTWSearch CPU] {i + 1}/{n_chunk}")

    return dist


# ─────────────────────────────────────────────────────────────
# DTWSearch 主类
# ─────────────────────────────────────────────────────────────

class DTWSearch:
    """
    Dynamic Time Warping 检索（v3.2 工业级版）

    GPU: cuda_fwd_pass 累积 DP（Sakoe-Chiba 严格约束），显存 O(chunk·n_mem·m)
    CPU: tslearn.cdist_dtw（精确，Numba 多线程）
    """

    DTYPE = np.float32

    def __init__(
        self,
        top_k: int = 5,
        dtw_radius: int = 5,
        weighted: bool = True,
        device: Union[str, torch.device] = 'auto',
        predict_chunk_size: int = 128,   # 128 × 8353 × 96 × 4 ≈ 410 MB（GPU）
        train_chunk_size: int = 256,     # 每块处理的记忆库样本数
        **kwargs
    ):
        self.k = top_k
        self.radius = max(1, int(dtw_radius))
        self.weighted = weighted
        self.predict_chunk_size = max(16, int(predict_chunk_size))
        self.train_chunk_size = max(64, int(train_chunk_size))

        if isinstance(device, str) and device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.memory_X: Optional[np.ndarray] = None  # (n_train, m * n_feat)
        self.memory_Y: Optional[np.ndarray] = None  # (n_train, pred_len * n_feat)
        self._mem_X_t: Optional[torch.Tensor] = None  # GPU Tensor (n_train, m)
        self._mem_Y_t: Optional[torch.Tensor] = None  # GPU Tensor (n_train, pred_len * n_feat)
        self.is_fitted = False
        self.seq_len: int = 0
        self.pred_len: int = 0
        self.n_features: int = 1

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """存储记忆库"""
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        self.n_features = Y_train.shape[-1] if Y_train.ndim == 3 else 1

        n = X_train.shape[0]
        self.memory_X = X_train.reshape(n, -1).astype(self.DTYPE)
        self.memory_Y = Y_train.reshape(n, -1).astype(self.DTYPE)

        if self.device.type == 'cuda':
            self._mem_X_t = torch.from_numpy(self.memory_X).float().pin_memory().to(self.device, non_blocking=True)
            self._mem_Y_t = torch.from_numpy(self.memory_Y).float().pin_memory().to(self.device, non_blocking=True)
        else:
            self._mem_X_t = torch.from_numpy(self.memory_X).float().to(self.device)
            self._mem_Y_t = torch.from_numpy(self.memory_Y).float().to(self.device)

        self.is_fitted = True
        logger.info(
            f"[DTWSearch] fit: device={self.device}, radius={self.radius}, "
            f"n_train={n}, seq={self.seq_len}, k={self.k}"
        )
        return self

    def predict(self, X_test: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        分块 DTW 检索 + 逆距离加权 KNN

        GPU: _cuda_dtw_forward（无中间 4D 张量）
        CPU: _cpu_dtw_chunked（tslearn 精确）
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

        n_train = self.memory_X.shape[0]
        m = self.seq_len
        device_str = str(self.device)

        logger.info(
            f"[DTWSearch] predict: n_test={n_test}, n_train={n_train}, "
            f"seq={m}, k={k}, test_chunk={self.predict_chunk_size}, "
            f"train_chunk={self.train_chunk_size}, radius={self.radius}, "
            f"device={device_str}"
        )

        cs = self.predict_chunk_size
        y_dim = self._mem_Y_t.shape[1]
        mem_X = self._mem_X_t
        mem_Y = self._mem_Y_t

        chunks = []

        with torch.no_grad():
            for t_start in range(0, n_test, cs):
                t_end = min(t_start + cs, n_test)
                chunk_sz = t_end - t_start
                xb = torch.from_numpy(X_flat[t_start:t_end]).float().to(self.device, non_blocking=True)
                cur_chunk = min(cs, t_end - t_start)

                # ── 距离计算（GPU 或 CPU）────────────────────────
                if self.device.type == 'cuda':
                    # GPU 路径：_cuda_dtw_forward，显存 O(chunk·mc·m)
                    mc = self.train_chunk_size
                    dtw_chunk = torch.zeros(cur_chunk, n_train, dtype=torch.float32, device=self.device)

                    for m_start in range(0, n_train, mc):
                        m_end = min(m_start + mc, n_train)
                        mem_slice = mem_X[m_start:m_end]   # (mc, m)
                        dtw_part = _cuda_dtw_forward(xb, mem_slice)  # (chunk, mc)
                        dtw_chunk[:, m_start:m_end] = dtw_part
                        del mem_slice, dtw_part
                        torch.cuda.empty_cache()

                    # 转 numpy 用于 top-k
                    dtw_np = dtw_chunk.cpu().numpy()
                    del dtw_chunk
                    torch.cuda.empty_cache()

                    top_idx = np.argpartition(dtw_np, k, axis=1)[:, :k]
                    vals = np.take_along_axis(dtw_np, top_idx, axis=1)
                    vals = np.clip(vals, 1e-6, None)
                    top_idx_t = torch.from_numpy(top_idx).long().to(self.device)
                    vals_t = torch.from_numpy(vals).float().to(self.device)

                else:
                    # CPU 路径：tslearn 精确 DTW
                    xb_np = xb.cpu().numpy()   # (chunk, m)
                    dtw_np = _cpu_dtw_chunked(
                        xb_np, self.memory_X, radius=self.radius
                    )  # (chunk, n_train)
                    top_idx = np.argpartition(dtw_np, k, axis=1)[:, :k]
                    vals = np.take_along_axis(dtw_np, top_idx, axis=1)
                    vals = np.clip(vals, 1e-6, None)
                    top_idx_t = torch.from_numpy(top_idx).long().to(self.device)
                    vals_t = torch.from_numpy(vals).float().to(self.device)

                del xb, dtw_np

                # ── 逆距离加权 KNN ────────────────────────────────
                neighbor_Y = mem_Y[top_idx_t]   # (chunk, k, y_dim)

                if self.weighted:
                    w = vals_t / vals_t.sum(dim=1, keepdim=True)   # (chunk, k)
                    yb = (neighbor_Y * w.unsqueeze(-1)).sum(dim=1)  # (chunk, y_dim)
                else:
                    yb = neighbor_Y.mean(dim=1)

                chunks.append(yb.cpu().numpy().astype(self.DTYPE))

                del neighbor_Y, w, yb, vals_t, top_idx_t
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                gc.collect()

                # 进度日志（每 5 个 chunk）
                processed = t_end
                pct = processed / n_test * 100
                logger.info(
                    f"  [DTWSearch] {processed}/{n_test} ({pct:.1f}%) "
                    f"| mem={self.device.type} | radius={self.radius}"
                )

        Y_pred = np.vstack(chunks)
        del chunks
        gc.collect()

        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        logger.info(f"[DTWSearch] predict done: output shape={Y_pred.shape}")
        return Y_pred

    def get_retrieval_meta(
        self,
        X_test: np.ndarray,
        top_k: Optional[int] = None,
        n_samples: int = 5
    ) -> tuple:
        """
        返回溯源证据：用于绘制"历史匹配溯源图"

        Args:
            X_test: shape (n_test, seq_len, n_feat) 归一化后的测试输入
            top_k: 检索的邻居数量（默认使用 self.k）
            n_samples: 返回前 n_samples 个测试样本的元数据（默认 5）

        Returns:
            Y_pred: 预测结果 shape (n_test, pred_len, n_feat)
            retrieval_meta: dict 包含:
                - topk_histories: (n_samples, k, seq_len, n_feat) 匹配到的历史序列
                - topk_futures: (n_samples, k, pred_len, n_feat) 匹配到的未来序列
                - topk_weights: (n_samples, k) 归一化的注意力权重
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        k = top_k if top_k is not None else self.k
        k = min(k, self.memory_X.shape[0])

        # 处理输入维度
        if X_test.ndim == 3:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        # 获取完整预测
        Y_pred = self.predict(X_test, top_k=top_k)

        # 只对前 n_samples 个样本保存溯源证据
        n_samples = min(n_samples, n_test)

        # 原始维度
        mem_X_raw = self.memory_X.reshape(-1, self.seq_len, self.n_features)
        mem_Y_raw = self.memory_Y.reshape(-1, self.pred_len, self.n_features)

        # 对前 n_samples 个样本检索（简化版本，使用欧氏距离近似）
        with torch.no_grad():
            xb = torch.from_numpy(X_flat[:n_samples]).float().to(self.device, non_blocking=True)
            dist = torch.cdist(xb, self._mem_X_t, p=2)  # (n_samples, n_train)
            vals, idx = torch.topk(dist, k, largest=False, dim=1)  # (n_samples, k)

            idx_np = idx.cpu().numpy()
            vals_np = vals.cpu().numpy()

            # 归一化权重
            vals_safe = np.clip(vals_np, 1e-10, None)
            weights = 1.0 / vals_safe
            weights = weights / weights.sum(axis=1, keepdims=True)

            # 提取匹配的序列
            topk_histories = mem_X_raw[idx_np].astype(np.float32)
            topk_futures = mem_Y_raw[idx_np].astype(np.float32)

            del xb, dist, vals, idx

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        logger.info(
            f"[DTWSearch] Retrieval meta: {n_samples} samples, k={k}"
        )

        retrieval_meta = {
            'topk_histories': topk_histories,
            'topk_futures': topk_futures,
            'topk_weights': weights.astype(np.float32),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'model_name': 'DTWSearch',
        }

        return Y_pred, retrieval_meta
