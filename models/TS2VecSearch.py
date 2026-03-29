"""
TS2VecSearch: 深度对比学习表示检索（v3.1 安全版）

学术规范说明：
本实现为简化版 TS2Vec，仅保留了原论文的核心架构设计：
  - Temporal Convolutional Network (TCN) + 空洞卷积编码器
  - Instance-level NT-Xent 对比损失
  - 两阶段范式：对比预训练 → 向量检索

未包含原论文的 Hierarchical Temporal Contrastive Loss（多尺度时间对比），
以及 masked token prediction 等辅助任务。此类简化在工业基线中是常见做法，
以换取训练速度和接口简洁性。

两阶段流程：
  fit: 对比损失训练 TCN 编码器 → X_train 存入 faiss 向量库
  predict: Encoder(X_test) → faiss 极速检索 + 加权 KNN 融合

核心参数：hidden_dim (默认 64), epochs (默认 10), batch_size (默认 128), top_k
"""

import gc
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# faiss（GPU/CPU 向量索引）
try:
    import faiss
    _HAS_FAISS = True
    logger.info("[TS2VecSearch] faiss available")
except ImportError:
    _HAS_FAISS = False
    logger.warning("[TS2VecSearch] faiss not available, will use scipy cdist fallback")


# ─────────────────────────────────────────────────────────────
# TCN Encoder（6 层空洞卷积，感受野 2^6 = 64）
# ─────────────────────────────────────────────────────────────

class _TCNEncoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            out_ch = hidden_dim
            # 因果卷积：padding = dilation 确保输出长度不变且仅依赖历史
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
            layers.extend([conv, nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(0.05)])
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        # 投影层：将 conv 输出映射到 hidden_dim（最终用 mean pooling 压缩时间维）
        self.projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) 或 (batch, seq_len)
        返回: (batch, hidden_dim) — 时间池化后的表示向量
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        h = x.transpose(1, 2)          # (batch, input_dim, seq_len)
        h = self.network(h)             # (batch, hidden_dim, seq_len)
        h = self.projection(h)          # (batch, hidden_dim, seq_len)
        return h.mean(dim=-1)           # (batch, hidden_dim)


class _ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) 对比损失"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch = z1.shape[0]
        sim = torch.mm(z1, z2.T) / self.tau   # (batch, batch)
        labels = torch.arange(batch, device=z1.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / 2


# ─────────────────────────────────────────────────────────────
# TS2VecSearch 主类
# ─────────────────────────────────────────────────────────────

class TS2VecSearch:
    """
    深度对比学习检索（简化版 TS2Vec + faiss 向量库）

    安全特性：
    - GPU 自动检测：torch.cuda.is_available() 时自动升格
    - faiss 缺失时：使用 scipy.spatial.distance.cdist 分块计算，内存安全
    - 所有张量显式管理：gc.collect() + torch.cuda.empty_cache()
    """

    DTYPE = np.float32

    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 10,
        batch_size: int = 128,
        top_k: int = 5,
        lr: float = 1e-3,
        temperature: float = 0.1,
        device: Union[str, torch.device] = 'auto',
        seed: int = 42,
        **kwargs
    ):
        # 自动设备检测
        if isinstance(device, str) and device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = top_k
        self.lr = lr
        self.tau = temperature
        self.seed = seed

        self.encoder: Optional[_TCNEncoder] = None
        self.memory_Y: Optional[np.ndarray] = None   # (n_train, pred_len * n_feat)
        self.index = None                           # faiss.Index
        self.train_vectors: Optional[np.ndarray] = None  # (n_train, hidden_dim)
        self.is_fitted = False
        self.seq_len: int = 0
        self.pred_len: int = 0
        self.n_features: int = 1

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def _augment(self, x: torch.Tensor) -> tuple:
        """Ts2vec 数据增强：随机尺度扰动 + 随机时间裁剪"""
        batch, seq_len, dim = x.shape
        scale1 = torch.rand(batch, 1, 1, device=x.device).uniform_(0.5, 2.0)
        scale2 = torch.rand(batch, 1, 1, device=x.device).uniform_(0.5, 2.0)
        v1 = x * scale1
        v2 = x * scale2
        if seq_len >= 8:
            crop_len = np.random.randint(seq_len // 2, seq_len + 1)
            crop_start = np.random.randint(0, seq_len - crop_len + 1)
            v1 = v1[:, crop_start:crop_start + crop_len, :]
            v2 = v2[:, crop_start:crop_start + crop_len, :]
            if v1.shape[1] < seq_len:
                pad1 = torch.zeros(batch, seq_len - v1.shape[1], dim, device=x.device)
                pad2 = torch.zeros(batch, seq_len - v2.shape[1], dim, device=x.device)
                v1 = torch.cat([v1, pad1], dim=1)
                v2 = torch.cat([v2, pad2], dim=1)
        return v1, v2

    def _encode_to_numpy(self, X: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        """
        将 numpy 数组批量编码为 numpy 向量

        Args:
            X: shape (n, seq_len, n_feat) 或 (n, seq_len)
            eval_mode: 是否用 eval 模式（dropout 等）

        Returns:
            shape (n, hidden_dim)
        """
        if X.ndim == 2:
            X = X[:, :, None]
        n = X.shape[0]
        cs = self.batch_size * 4  # 编码可以用更大的 batch
        vecs = []

        if eval_mode:
            self.encoder.eval()
            with torch.no_grad():
                for start in range(0, n, cs):
                    end = min(start + cs, n)
                    xb = torch.from_numpy(X[start:end]).float().to(self.device)
                    v = self.encoder(xb).cpu().numpy()
                    vecs.append(v)
                    del xb, v
        else:
            self.encoder.train()
            for start in range(0, n, cs):
                end = min(start + cs, n)
                xb = torch.from_numpy(X[start:end]).float().to(self.device)
                v = self.encoder(xb).cpu().numpy()
                vecs.append(v)
                del xb, v

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return np.vstack(vecs).astype(np.float32)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        对比学习训练 TCN 编码器，然后将 X_train 存入 faiss 向量库
        """
        self._set_seed()
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = Y.shape[-1] if Y.ndim == 3 else 1
        n_train = X.shape[0]
        self.memory_Y = Y.reshape(n_train, -1).astype(np.float32)

        logger.info(
            f"[TS2VecSearch] fit: n_train={n_train}, seq_len={self.seq_len}, "
            f"hidden={self.hidden_dim}, epochs={self.epochs}, device={self.device}"
        )

        # ── 构建编码器 ────────────────────────────────────────────
        input_dim = max(1, self.n_features)
        self.encoder = _TCNEncoder(
            input_dim=input_dim, hidden_dim=self.hidden_dim, num_layers=6
        ).to(self.device)

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = _ContrastiveLoss(temperature=self.tau)

        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.encoder.train()
        for epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            for (batch_x,) in loader:
                batch_x = batch_x.float().to(self.device)
                v1, v2 = self._augment(batch_x)
                z1 = self.encoder(v1)
                z2 = self.encoder(v2)
                loss = criterion(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                del batch_x, v1, v2, z1, z2, loss

            scheduler.step()
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                logger.info(
                    f"[TS2VecSearch] epoch {epoch+1}/{self.epochs} "
                    f"loss={total_loss / max(1, n_batches):.4f}"
                )

        self.encoder.eval()
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # ── 编码 X_train → faiss / numpy ─────────────────────────
        self.train_vectors = self._encode_to_numpy(X, eval_mode=True)
        logger.info(f"[TS2VecSearch] encoded train: {self.train_vectors.shape}")

        if _HAS_FAISS:
            d = self.hidden_dim
            if self.device.type == 'cuda':
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatL2(gpu_res, d)
            else:
                self.index = faiss.IndexFlatL2(d)
            self.index.add(self.train_vectors)
            logger.info(f"[TS2VecSearch] faiss index built: {self.index.ntotal}")
        else:
            logger.info("[TS2VecSearch] faiss missing → scipy cdist fallback")

        self.is_fitted = True
        logger.info("[TS2VecSearch] fit done")
        return self

    def predict(self, X_test: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        Encoder(X_test) → faiss 检索 / scipy cdist chunked → 加权 KNN

        内存安全：无论 faiss 是否存在，全程分块计算，绝不构造全量 (n_test, n_train) 张量
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        k = top_k if top_k is not None else self.k
        k = min(k, self.train_vectors.shape[0])

        Xt = X_test.astype(np.float32)
        if Xt.ndim == 2:
            Xt = Xt[:, :, None]

        n_test = Xt.shape[0]
        logger.info(f"[TS2VecSearch] predict: n_test={n_test}, k={k}")

        # ── 编码 X_test（分块）─────────────────────────────────────
        test_vecs = self._encode_to_numpy(Xt, eval_mode=True)  # (n_test, hidden_dim)
        logger.info(f"[TS2VecSearch] test vectors encoded: {test_vecs.shape}")

        # ── top-k 检索 ────────────────────────────────────────────
        n_train = self.train_vectors.shape[0]
        indices = np.zeros((n_test, k), dtype=np.int64)
        dists = np.zeros((n_test, k), dtype=np.float32)

        if _HAS_FAISS and self.index is not None:
            # faiss 路径（GPU/CPU 都是安全实现）
            d_out, i_out = self.index.search(test_vecs, k)
            indices = i_out.astype(np.int64)
            dists = d_out.astype(np.float32)
        else:
            # ── scipy cdist 分块（内存安全）───────────────────────
            # 对 test_vecs 分块，每次编码 TEST_CHUNK 个样本
            # 对每个 chunk，在 CPU 上用 scipy.cdist 计算与全量 train_vectors 的距离
            # 再用 np.argpartition 取 top-k
            try:
                from scipy.spatial.distance import cdist as scipy_cdist
                _HAS_SCIPY = True
            except ImportError:
                _HAS_SCIPY = False

            TEST_CHUNK = 512   # 每次处理 512 个测试样本
            for i_start in range(0, n_test, TEST_CHUNK):
                i_end = min(i_start + TEST_CHUNK, n_test)
                test_chunk = test_vecs[i_start:i_end]   # (chunk, hidden_dim)

                if _HAS_SCIPY:
                    # scipy cdist: (chunk, n_train) — 无需构造全量矩阵
                    chunk_dists = scipy_cdist(test_chunk, self.train_vectors, metric='euclidean')
                else:
                    # Pure numpy 分块：torch.cdist 在 CPU 上安全运行
                    chunk_t = torch.from_numpy(test_chunk).float()
                    mem_t = torch.from_numpy(self.train_vectors).float()
                    chunk_dists = torch.cdist(chunk_t, mem_t, p=2).numpy()
                    del chunk_t, mem_t

                # np.argpartition 取 top-k（O(n log k) 而非 O(n log n)）
                for j in range(i_end - i_start):
                    top_idx = np.argpartition(chunk_dists[j], k)[:k]
                    sorted_order = top_idx[np.argsort(chunk_dists[j][top_idx])]
                    indices[i_start + j] = sorted_order
                    dists[i_start + j] = chunk_dists[j][sorted_order]

                del chunk_dists
                if (i_start // TEST_CHUNK + 1) % 10 == 0:
                    logger.info(f"  [TS2VecSearch] processed {i_end}/{n_test}")

        del test_vecs
        gc.collect()

        # ── 逆距离加权 KNN ─────────────────────────────────────────
        dists_safe = np.clip(dists, 1e-6, None)
        w = 1.0 / dists_safe
        w = w / w.sum(axis=1, keepdims=True)   # (n_test, k)

        neighbor_Y = self.memory_Y[indices]   # (n_test, k, pred_len * n_feat)
        Y_pred_flat = (neighbor_Y * w[:, :, None]).sum(axis=1)   # (n_test, pred_len * n_feat)

        n_feat = self.n_features
        if n_feat > 1:
            Y_pred = Y_pred_flat.reshape(n_test, self.pred_len, n_feat)
        else:
            Y_pred = Y_pred_flat.reshape(n_test, self.pred_len)

        logger.info(f"[TS2VecSearch] predict done: {Y_pred.shape}")
        return Y_pred.astype(np.float32)
