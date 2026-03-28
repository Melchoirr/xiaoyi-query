"""
TS2VecSearch: 深度对比学习表示检索

严格遵循 Ts2vec: Towards Universal Representation of Time Series (AAAI 2022) 核心思想。

学术规范：
- 自监督对比编码器：Temporal Convolutional Network (TCN) + 空洞卷积
- 两阶段训练：对比损失（Contrastive Loss）训练编码器 → faiss 向量检索
- fit 阶段：训练 TCN 编码器，将 X_train 转化为密集向量
- predict 阶段：用 faiss.IndexFlatL2 极速检索 + 加权 KNN 融合

核心参数：hidden_dim (默认 64), epochs (默认 10), batch_size (默认 128), top_k

架构：
  Input (seq_len) → Dilated Conv Block × 6 (感受野 2^6=64)
  → Temporal Pooling → dense_vec (hidden_dim)

Usage:
    from models.TS2VecSearch import TS2VecSearch
    model = TS2VecSearch(hidden_dim=64, epochs=10, batch_size=128, top_k=5)
    model.fit(X_train_norm, Y_train_norm)   # 训练对比编码器
    Y_pred = model.predict(X_test_norm)      # faiss 检索 + KNN
"""

import gc
import logging
import os
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
    logger.warning("[TS2VecSearch] faiss not available, using np argpartition")


# ─────────────────────────────────────────────────────────────
# TCN Encoder（空洞卷积编码器）
# ─────────────────────────────────────────────────────────────

class _TCNEncoder(nn.Module):
    """
    轻量级 TCN 编码器
    6 层空洞卷积，感受野 = 2^6 = 64（可覆盖常见 seq_len）
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            out_ch = hidden_dim
            pad = dilation  # 因果卷积补零
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation)
            layers.append(conv)
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)  # 投影到 hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) 或 (batch, seq_len)
        返回: (batch, hidden_dim) — 时间池化后的表示向量
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Conv1d 需要 (batch, channels, seq_len)
        h = x.transpose(1, 2)   # (batch, input_dim, seq_len)
        h = self.network(h)       # (batch, hidden_dim, seq_len)
        # 因果卷积导致序列末尾仍有有效信号，取最后时刻
        h = self.projection(h)   # (batch, hidden_dim, seq_len)
        # 时间池化：mean pooling over valid region
        out = h.mean(dim=-1)     # (batch, hidden_dim)
        return out


class _ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) 对比损失"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (batch, hidden_dim) — 同一 batch 的两个增强视图
        """
        batch = z1.shape[0]
        # 正样本对：对角线
        sim = torch.mm(z1, z2.T) / self.tau  # (batch, batch)
        labels = torch.arange(batch, device=z1.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / 2


# ─────────────────────────────────────────────────────────────
# TS2VecSearch 主类
# ─────────────────────────────────────────────────────────────

class TS2VecSearch:
    """
    TS2Vec 深度对比学习检索

    严格遵循 Ts2vec (AAAI 2022)：
    - fit: 对比损失训练 TCN 编码器 → 将 X_train 存入 faiss 向量库
    - predict: Encoder(X_test) → faiss 检索 top_k → 加权 KNN 融合 Y_train

    Args:
        hidden_dim: 编码向量维度（默认 64）
        epochs: 对比学习训练轮数（默认 10）
        batch_size: 训练批大小（默认 128）
        top_k: 检索近邻数（默认 5）
        lr: 学习率（默认 1e-3）
        temperature: 对比损失温度（默认 0.1）
        device: 计算设备（默认 'cpu'）
        seed: 随机种子（默认 42）
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
        device: Union[str, torch.device] = 'cpu',
        seed: int = 42,
        **kwargs
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = top_k
        self.lr = lr
        self.tau = temperature
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.seed = seed

        self.encoder: Optional[_TCNEncoder] = None
        self.memory_Y: Optional[np.ndarray] = None   # (n_train, pred_len * n_feat)
        self.index = None                           # faiss.IndexFlatL2
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
        """
        Ts2vec 数据增强：随机裁剪 + 幅度变换
        返回两个增强视图 (view1, view2): shape (batch, seq_len, 1)
        """
        batch, seq_len, dim = x.shape
        # 视图 1：随机尺度扰动
        scale1 = torch.FloatTensor(batch, 1, 1).uniform_(0.5, 2.0).to(x.device)
        v1 = x * scale1
        # 视图 2：随机尺度扰动（不同噪声）+ 随机裁剪
        scale2 = torch.FloatTensor(batch, 1, 1).uniform_(0.5, 2.0).to(x.device)
        v2 = x * scale2
        # 随机时间裁剪
        if seq_len >= 8:
            crop_len = np.random.randint(seq_len // 2, seq_len + 1)
            crop_start = np.random.randint(0, seq_len - crop_len + 1)
            v1 = v1[:, crop_start:crop_start + crop_len, :]
            v2 = v2[:, crop_start:crop_start + crop_len, :]
            # 不足 seq_len 的部分 pad（0 填充）
            if v1.shape[1] < seq_len:
                pad1 = torch.zeros(batch, seq_len - v1.shape[1], dim, device=x.device)
                pad2 = torch.zeros(batch, seq_len - v2.shape[1], dim, device=x.device)
                v1 = torch.cat([v1, pad1], dim=1)
                v2 = torch.cat([v2, pad2], dim=1)
        return v1, v2

    def _encode_batch(self, x: np.ndarray) -> np.ndarray:
        """将 numpy 数组批量编码为向量"""
        x_t = torch.from_numpy(x).float().to(self.device)
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(-1)
        with torch.no_grad():
            vecs = self.encoder(x_t).cpu().numpy()
        return vecs

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        对比学习训练 TCN 编码器，然后将 X_train 存入 faiss 向量库

        Args:
            X_train: shape (n_train, seq_len, n_feat) 或 (n_train, seq_len)
            Y_train: shape (n_train, pred_len, n_feat) 或 (n_train, pred_len)
        """
        self._set_seed()
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = Y.shape[-1] if Y.ndim == 3 else 1

        n_train = X.shape[0]
        self.memory_Y = Y.reshape(n_train, -1).astype(np.float32)

        # 展平为 (n_train, seq_len, 1) — encoder 每次处理一个特征维度
        # 对多变量：先对各维独立编码再拼接
        if self.n_features == 1:
            X_3d = X[:, :, None]  # (n_train, seq_len, 1)
        else:
            X_3d = X  # (n_train, seq_len, n_feat)

        logger.info(
            f"[TS2VecSearch] fit: n_train={n_train}, seq_len={self.seq_len}, "
            f"hidden_dim={self.hidden_dim}, epochs={self.epochs}, device={self.device}"
        )

        # ── 构建 TCN 编码器 ───────────────────────────────────────
        input_dim = max(1, self.n_features)
        self.encoder = _TCNEncoder(
            input_dim=input_dim, hidden_dim=self.hidden_dim, num_layers=6
        ).to(self.device)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = _ContrastiveLoss(temperature=self.tau)

        # DataLoader
        dataset = TensorDataset(torch.from_numpy(X_3d))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.encoder.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            for (batch_x,) in loader:
                batch_x = batch_x.float().to(self.device)  # (batch, seq_len, input_dim)

                # 两个增强视图
                v1, v2 = self._augment(batch_x)

                # 编码
                z1 = self.encoder(v1)   # (batch, hidden_dim)
                z2 = self.encoder(v2)  # (batch, hidden_dim)

                # 对比损失
                loss = criterion(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                logger.info(
                    f"[TS2VecSearch] epoch {epoch+1}/{self.epochs} "
                    f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
                )

        self.encoder.eval()

        # ── 将 X_train 全部编码存入向量库 ─────────────────────────
        vecs = self._encode_batch(X_3d)  # (n_train, hidden_dim)
        self.train_vectors = vecs.astype(np.float32)

        if _HAS_FAISS:
            d = self.hidden_dim
            if self.device.type == 'cuda':
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatL2(gpu_res, d)
            else:
                self.index = faiss.IndexFlatL2(d)
            self.index.add(self.train_vectors)
            logger.info(f"[TS2VecSearch] faiss index built: {self.index.ntotal} vectors")
        else:
            logger.info("[TS2VecSearch] faiss not available, using np argpartition")

        self.is_fitted = True
        logger.info("[TS2VecSearch] fit done")
        return self

    def predict(self, X_test: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        Encoder(X_test) → faiss 检索 → 加权 KNN 融合 Y_train

        Args:
            X_test: shape (n_test, seq_len, n_feat) 或 (n_test, seq_len)
            top_k: 覆盖默认的 k

        Returns:
            shape (n_test, pred_len, n_feat) 或 (n_test, pred_len)
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

        # 编码 X_test
        test_vecs = self._encode_batch(Xt)  # (n_test, hidden_dim)

        if _HAS_FAISS and self.index is not None:
            dists, indices = self.index.search(test_vecs, k)  # (n_test, k)
        else:
            # fallback: np.argpartition
            n_train = self.train_vectors.shape[0]
            dists_mat = np.linalg.norm(
                test_vecs[:, None, :] - self.train_vectors[None, :, :], axis=2
            )  # (n_test, n_train)
            indices = np.zeros((n_test, k), dtype=np.int64)
            dists = np.zeros((n_test, k), dtype=np.float32)
            for i in range(n_test):
                top_idx = np.argpartition(dists_mat[i], k)[:k]
                indices[i] = top_idx
                dists[i] = dists_mat[i, top_idx]

        # 逆距离加权 KNN
        dists_safe = np.where(dists < 1e-8, 1e-8, dists)
        w = 1.0 / dists_safe
        w = w / w.sum(axis=1, keepdims=True)  # (n_test, k)

        # 融合 Y
        neighbor_Y = self.memory_Y[indices]  # (n_test, k, pred_len * n_feat)
        w_expanded = w[:, :, None]           # (n_test, k, 1)
        Y_pred_flat = (neighbor_Y * w_expanded).sum(axis=1)  # (n_test, pred_len * n_feat)

        y_dim_total = self.pred_len * self.n_features
        Y_pred = Y_pred_flat.reshape(n_test, self.pred_len, self.n_features) \
            if self.n_features > 1 else Y_pred_flat.reshape(n_test, self.pred_len)

        del test_vecs
        gc.collect()

        logger.info(f"[TS2VecSearch] predict done: output shape={Y_pred.shape}")
        return Y_pred.astype(np.float32)
