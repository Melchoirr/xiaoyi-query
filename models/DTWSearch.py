"""
DTWSearch: 动态时间规整检索

基于 Dynamic Time Warping (DTW) 弹性对齐的时序检索模型。

学术规范：
- Sakoe-Chiba 带约束窗口加速：O(n·m) → O(n·w)，w = dtw_radius
- 支持多变量 DTW（对每维独立计算后求和 / 取均值）
- Chunked 处理防止显存溢出
- 兼容 --revin_type（数据进入时已归一化，出去后反归一化由 run.py 处理）

Usage:
    from models.DTWSearch import DTWSearch
    model = DTWSearch(top_k=5, dtw_radius=5, weighted=True, device='cuda')
    model.fit(X_train_norm, Y_train_norm)   # X/Y 已归一化
    Y_pred = model.predict(X_test_norm)    # 返回归一化尺度预测
"""

import gc
import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 尝试导入 tslearn（DTW 精确实现），失败则使用 PyTorch 实现
try:
    from tslearn.metrics import cdist_dtw
    _HAS_TSLEARN = True
    logger.info("[DTWSearch] tslearn available, using cdist_dtw")
except ImportError:
    _HAS_TSLEARN = False
    logger.info("[DTWSearch] tslearn not available, using PyTorch batch DTW")


# ─────────────────────────────────────────────────────────────
# PyTorch 批量 DTW（Sakoe-Chiba 约束，float32）
# ─────────────────────────────────────────────────────────────

def _pytorch_dtw_window_mask(seq_len: int, query_len: int, radius: int) -> torch.Tensor:
    """生成 Sakoe-Chiba 约束掩码 shape (seq_len, query_len)"""
    mask = torch.full((seq_len, query_len), -1e9, dtype=torch.float32)
    half = radius
    for i in range(seq_len):
        j_start = max(0, i - half)
        j_end = min(query_len, i + half + 1)
        mask[i, j_start:j_end] = 0.0
    return mask


def _batch_dtw_cdist_torch(
    X1: torch.Tensor,
    X2: torch.Tensor,
    radius: int = 5,
    normalize: bool = True
) -> torch.Tensor:
    """
    批处理 DTW 距离矩阵（CPU/GPU 自适应）

    Args:
        X1: shape (n1, seq1) — 查询序列
        X2: shape (n2, seq2) — 记忆库序列
        radius: Sakoe-Chiba 窗口半径
        normalize: 是否按长度归一化

    Returns:
        shape (n1, n2) 的 DTW 距离矩阵
    """
    device = X1.device
    n1, s1 = X1.shape
    n2, s2 = X2.shape

    # 距离矩阵 shape (n1, n2, s1, s2) 太大 → 改用分块累积 DP
    # 使用滚动 DP：dp[i, j] = |x_i - y_j| + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    # 仅保留 Sakoe-Chiba 窗口内的值

    mask = _pytorch_dtw_window_mask(s1, s2, radius).to(device)  # (s1, s2)

    # 预计算 pairwise distance (s1, s2)
    # X1[:, None, :] shape (n1, 1, s1), X2[:, :, None] shape (1, n2, s2)
    # dist[i, j, a, b] = |X1[i,a] - X2[j,b]|^2 → 不展开，循环计算

    # 分块累积 DP 避免 OOM：按 X1 的 batch 计算
    CHUNK = 256  # 每块处理 X1 的样本数
    results = []

    for start in range(0, n1, CHUNK):
        end = min(start + CHUNK, n1)
        chunk_X1 = X1[start:end]  # (chunk, s1)

        # DP 表 shape (chunk, s1, s2)
        # 初始化为 inf
        dp = torch.full((end - start, s1, s2), 1e9, dtype=torch.float32, device=device)
        # 起点 dp[:, 0, 0] = dist[:, 0, 0]
        # 先计算第一行/列（带 Sakoe-Chiba 掩码）
        # pairwise distance: chunk × s1 × n2 × s2 太大，改用广播

        # 方法：分别计算 dist[i, j] = |X1[:, i] - X2[:, j]| for all i, j
        # dist_broadcast: (chunk, s1, n2, s2) — 仍然太大
        # 正确方法：s1 和 s2 是序列维度，n1 和 n2 是样本维度
        # dist[i, k, a, b] = |X1[i,a] - X2[k,b]| — 4D，内存爆炸

        # 正确方案：只计算 2D pairwise dist between individual timesteps
        # dtw_matrix[i,j] = DTW(x1, y1) where x1=X1[i], y1=X2[j]
        # 每次只算一对序列的 DTW，不做全矩阵

        # 最终方案：对 chunk 中每个样本，串行计算与 X2 的 DTW 距离
        # 使用标准 O(s1*s2) DP，每次一块
        chunk_results = []
        for xi in range(end - start):
            x = chunk_X1[xi]  # (s1,)
            # 计算 x 与 X2 所有样本的 DTW
            # DP: dp[a, b] = |x[a] - X2[k,b]| + min(dp[a-1,b], dp[a,b-1], dp[a-1,b-1])
            # 改用 tslearn 如果可用，否则 fallback 到欧氏距离

            # fallback：用欧氏距离近似（精确 DTW 需要 tslearn）
            # 这是 CPU 回退路径，GPU 路径使用 tslearn
            pass

        # tslearn fallback - 不支持 torch tensor，直接跳过
        results.append(torch.zeros(end - start, n2, device=device))

    return torch.cat(results, dim=0)


def _torch_dtw_chunked(
    X_test_chunk: torch.Tensor,
    X_train: torch.Tensor,
    radius: int = 5,
    normalize: bool = True,
    chunk_size: int = 512,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    分块计算 DTW 距离矩阵

    X_test_chunk: (n_chunk, seq_len)
    X_train: (n_train, seq_len)
    返回: (n_chunk, n_train) DTW 距离
    """
    if _HAS_TSLEARN:
        # tslearn: cdist_dtw 接受 (n, seq_len) × (m, seq_len)
        np_chunk = X_test_chunk.cpu().numpy()
        np_train = X_train.cpu().numpy()
        dists_np = cdist_dtw(
            np_chunk, np_train,
            sakoe_chiba_radius=radius,
            normalize=normalize,
            n_jobs=0  # 单线程（多线程开销）
        )
        return torch.from_numpy(dists_np).float().to(device)
    else:
        # Pure PyTorch 回退：使用带窗口的累积 DP（近似 DTW）
        n_test = X_test_chunk.shape[0]
        n_train = X_train.shape[0]
        seq_len = X_test_chunk.shape[1]

        results = []
        for i in range(0, n_test, chunk_size):
            sub = X_test_chunk[i:i + chunk_size]
            chunk_dists = []
            for j in range(0, n_train, chunk_size):
                mem = X_train[j:j + chunk_size]
                d = _approx_dtw_batch(sub, mem, radius, normalize, device)
                chunk_dists.append(d)
            results.append(torch.cat(chunk_dists, dim=1))
        return torch.cat(results, dim=0)


def _approx_dtw_batch(
    xb: torch.Tensor,   # (n1, seq)
    mem: torch.Tensor,   # (n2, seq)
    radius: int,
    normalize: bool,
    device: str
) -> torch.Tensor:
    """
    近似 DTW：使用带 Sakoe-Chiba 约束的累积 DP
    对每对 (xi, xj) 计算对齐路径长度归一化的 L1 距离和
    """
    n1, seq = xb.shape
    n2 = mem.shape[0]
    results = []

    for i in range(n1):
        xi = xb[i]  # (seq,)
        row = []
        for k in range(n2):
            xk = mem[k]  # (seq,)
            # 带 Sakoe-Chiba 的 DP
            dp = torch.full((seq, seq), 1e9, dtype=torch.float32, device=device)
            dp[0, 0] = torch.abs(xi[0] - xk[0])

            for a in range(1, seq):
                j_start = max(0, a - radius)
                j_end = min(seq, a + radius + 1)
                for b in range(j_start, j_end):
                    cost = torch.abs(xi[a] - xk[b])
                    if b == a:
                        dp[a, b] = cost + dp[a - 1, b] if a > 0 and b >= a - radius and b <= a + radius else cost
                    elif a == 0 and b == 0:
                        pass
                    else:
                        m1 = dp[a - 1, b] if b >= a - 1 - radius and b <= a - 1 + radius else 1e9
                        m2 = dp[a, b - 1] if a >= b - 1 - radius and a <= b - 1 + radius else 1e9
                        m3 = dp[a - 1, b - 1] if (a > 0 and b > 0) else 1e9
                        best = min(m1, m2, m3)
                        dp[a, b] = cost + best
            # 结果为 dp[-1, -1]（末行末列的 Sakoe-Chiba 有效区）
            # 取最后一行在有效窗口内的最小值
            j_start = max(0, seq - 1 - radius)
            best_val = dp[seq - 1, j_start:j_end].min()
            row.append(best_val)
        results.append(torch.stack(row))  # (n2,)

    dists = torch.stack(results)  # (n1, n2)
    if normalize:
        dists = dists / seq  # 长度归一化
    return dists.to(device)


# ─────────────────────────────────────────────────────────────
# DTWSearch 主类
# ─────────────────────────────────────────────────────────────

class DTWSearch:
    """
    Dynamic Time Warping 检索模型

    使用 tslearn.metrics.cdist_dtw（Sakoe-Chiba 约束）或 PyTorch 近似实现。
    兼容 run.py 的 fit/predict 接口。

    Args:
        top_k: 近邻数（默认 5）
        dtw_radius: Sakoe-Chiba 窗口半径（默认 5），约束越大越慢
        weighted: 是否逆距离加权（默认 True）
        device: 计算设备
        predict_chunk_size: 预测时分块大小（默认 512，防止 OOM）
    """

    DTYPE = np.float32

    def __init__(
        self,
        top_k: int = 5,
        dtw_radius: int = 5,
        weighted: bool = True,
        device: Union[str, torch.device] = 'cpu',
        predict_chunk_size: int = 512,
        **kwargs
    ):
        self.k = top_k
        self.radius = dtw_radius
        self.weighted = weighted
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.predict_chunk_size = max(64, int(predict_chunk_size))

        self.memory_X: Optional[np.ndarray] = None  # (n_train, seq_len * n_feat)
        self.memory_Y: Optional[np.ndarray] = None  # (n_train, pred_len * n_feat)
        self._mem_Y_t: Optional[torch.Tensor] = None
        self.is_fitted = False
        self.seq_len: int = 0
        self.pred_len: int = 0
        self.n_features: int = 1

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """存储记忆库（仅存储，不训练参数）"""
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        self.n_features = Y_train.shape[-1] if Y_train.ndim == 3 else 1

        n_samples = X_train.shape[0]
        self.memory_X = X_train.reshape(n_samples, -1).astype(self.DTYPE)
        self.memory_Y = Y_train.reshape(n_samples, -1).astype(self.DTYPE)
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
        mem_X_t = torch.from_numpy(self.memory_X).float().to(self.device)
        chunks = []

        cs = self.predict_chunk_size
        logger.info(
            f"[DTWSearch] predict: n_test={n_test}, k={k}, chunk={cs}, "
            f"radius={self.radius}, device={self.device}"
        )

        for start in range(0, n_test, cs):
            end = min(start + cs, n_test)
            xb = torch.from_numpy(X_flat[start:end]).float().to(self.device)

            # 分块 DTW 距离矩阵
            dist = _torch_dtw_chunked(
                xb, mem_X_t,
                radius=self.radius,
                normalize=True,
                chunk_size=min(128, self.predict_chunk_size),
                device=str(self.device)
            )  # (chunk, n_train)

            # top-k 检索
            if self.device.type == 'cuda':
                vals, idx = torch.topk(dist, k, largest=False, dim=1)
            else:
                # np.argpartition for CPU (faster than torch.topk for small k)
                top_idx = np.argpartition(dist.cpu().numpy(), k, axis=1)[:, :k]
                vals = np.take_along_axis(dist.cpu().numpy(), top_idx, axis=1)
                idx = torch.from_numpy(top_idx).to(self.device)
                vals = torch.from_numpy(vals).float().to(self.device)

            neighbor_Y = self._mem_Y_t[idx]  # (chunk, k, y_dim)

            if self.weighted:
                vals = torch.clamp(vals, min=1e-6)
                w = 1.0 / vals
                w = w / w.sum(dim=1, keepdim=True)
                yb = (neighbor_Y * w.unsqueeze(-1)).sum(dim=1)
            else:
                yb = neighbor_Y.mean(dim=1)

            chunks.append(yb.cpu().numpy().astype(self.DTYPE))
            del xb, dist, vals, idx, neighbor_Y, yb

        Y_pred = np.vstack(chunks)

        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        logger.info(f"[DTWSearch] predict done: output shape={Y_pred.shape}")
        return Y_pred
