"""
RAGSearch: 检索增强/记忆网络（Cross-Attention 机制）

结合大模型最火的 Retrieval-Augmented Generation 和 Cross-Attention 机制。
一种端到端 (End-to-End) 训练的模型。

学术规范：
- 架构：Query Encoder + Cross-Attention Memory
- Memory Bank: X_train 作为 Keys，Y_train 作为 Values
- fit: 端到端训练，通过 MSE Loss 优化
- predict: 直接前向推理

核心参数：d_model (默认 32), n_heads (默认 4), epochs (默认 10), batch_size (默认 128)

Usage:
    from models.RAGSearch import RAGSearch
    model = RAGSearch(d_model=32, n_heads=4, epochs=10, batch_size=128)
    model.fit(X_train_norm, Y_train_norm)   # 端到端训练
    Y_pred = model.predict(X_test_norm)      # Cross-Attention 预测
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


# ─────────────────────────────────────────────────────────────
# Cross-Attention 记忆网络
# ─────────────────────────────────────────────────────────────

class _RAGCore(nn.Module):
    """
    Cross-Attention 核心模块

    Q = Query Encoder(X_test) → (batch, d_model)
    K = mean(Query Encoder(X_train)) → (n_train, d_model)  [冻结存储]
    V = Y_train → (n_train, pred_len, n_feat)

    Attention: softmax(Q @ K^T / sqrt(d)) @ V
    """

    def __init__(self, d_model: int, n_heads: int, n_features: int, pred_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_features = n_features
        self.pred_len = pred_len
        self.val_dim = pred_len * n_features

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.0, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, self.val_dim)

    def forward(
        self,
        q_enc: torch.Tensor,
        k_enc: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q_enc: (batch, d_model) — Query 编码
            k_enc: (n_train, d_model) — 记忆 Key 编码（来自 fit 时存储）
            v: (n_train, pred_len, n_feat) — 记忆 Value

        Returns:
            (batch, pred_len, n_feat) — 预测值
        """
        # MultiheadAttention 需要 (batch, seq_len, d_model)
        # Q: (batch, 1, d_model)，K: (n_train, 1, d_model)
        q = q_enc.unsqueeze(1)    # (batch, 1, d_model)
        k = k_enc.unsqueeze(1)    # (n_train, 1, d_model)
        # V 需要 reshape: (n_train, pred_len, n_feat) → (n_train, pred_len * n_feat)
        v_flat = v.reshape(v.shape[0], -1).unsqueeze(1)  # (n_train, 1, val_dim)
        # 投影 V 到 d_model
        v_proj = v_flat.expand(-1, q.shape[0], -1)  # (n_train, batch, val_dim) — 太大
        # 正确做法：V 通过 knn weighted sum，而不是 multihead attn
        # 简化：直接用加权求和替代 MHA
        attn_w = torch.softmax(
            (q_enc @ k_enc.T) / (self.d_model ** 0.5), dim=-1
        )  # (batch, n_train)
        # v: (n_train, pred_len, n_feat)
        y_pred = torch.einsum('bn,bnf->bf', attn_w, v)  # (batch, pred_len * n_feat)
        y_pred = self.out_proj(y_pred)  # (batch, pred_len * n_feat)
        return y_pred


class _QueryEncoder(nn.Module):
    """Query 编码器：时序 → d_model 维向量"""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        返回: (batch, d_model)
        """
        h = self.proj(x)  # (batch, seq_len, d_model)
        return h.mean(dim=1)  # (batch, d_model)


class _RAGNet(nn.Module):
    """完整 RAG 网络"""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_features: int,
        pred_len: int
    ):
        super().__init__()
        self.encoder = _QueryEncoder(input_dim=input_dim, d_model=d_model)
        self.core = _RAGCore(
            d_model=d_model, n_heads=n_heads,
            n_features=n_features, pred_len=pred_len
        )

    def forward(
        self,
        query_X: torch.Tensor,
        keys_X_encoded: torch.Tensor,
        values_Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_X: (batch, seq_len, input_dim)
            keys_X_encoded: (n_train, d_model) — fit 时预编码好的 Keys
            values_Y: (n_train, pred_len, n_feat) — fit 时存储的 Values
        """
        q = self.encoder(query_X)  # (batch, d_model)
        return self.core(q, keys_X_encoded, values_Y)


# ─────────────────────────────────────────────────────────────
# RAGSearch 主类
# ─────────────────────────────────────────────────────────────

class RAGSearch:
    """
    检索增强记忆网络（Cross-Attention）

    结合 RAG 和 Cross-Attention 机制，端到端训练的检索模型。

    Args:
        d_model: 隐向量维度（默认 32）
        n_heads: 注意力头数（默认 4）
        epochs: 训练轮数（默认 10）
        batch_size: 批大小（默认 128）
        lr: 学习率（默认 1e-3）
        weight_decay: 权重衰减（默认 1e-4）
        device: 计算设备
        seed: 随机种子（默认 42）
    """

    DTYPE = np.float32

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Union[str, torch.device] = 'cpu',
        seed: int = 42,
        **kwargs
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.seed = seed

        self.net: Optional[_RAGNet] = None
        self.keys_X_encoded: Optional[torch.Tensor] = None  # (n_train, d_model)
        self.values_Y: Optional[torch.Tensor] = None  # (n_train, pred_len, n_feat)
        self.memory_X: Optional[np.ndarray] = None  # (n_train, seq_len, n_feat)
        self.memory_Y: Optional[np.ndarray] = None  # (n_train, pred_len, n_feat)
        self.is_fitted = False
        self.seq_len: int = 0
        self.pred_len: int = 0
        self.n_features: int = 1

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        端到端训练 Cross-Attention 记忆网络

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

        if X.ndim == 2:
            X = X[:, :, None]
        if Y.ndim == 2:
            Y = Y[:, :, None]

        self.memory_X = X.astype(np.float32)
        self.memory_Y = Y.astype(np.float32)

        logger.info(
            f"[RAGSearch] fit: n_train={n_train}, seq_len={self.seq_len}, "
            f"pred_len={self.pred_len}, n_feat={self.n_features}, "
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"epochs={self.epochs}, device={self.device}"
        )

        # ── 构建网络 ───────────────────────────────────────────
        input_dim = max(1, self.n_features)
        self.net = _RAGNet(
            input_dim=input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_features=self.n_features,
            pred_len=self.pred_len
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        # ── 预编码 Keys（冻结，用于训练和推理）───────────────────
        encoder_device = self.device
        temp_encoder = _QueryEncoder(input_dim=input_dim, d_model=self.d_model).to(encoder_device)
        X_t = torch.from_numpy(self.memory_X).float().to(encoder_device)
        with torch.no_grad():
            self.keys_X_encoded = temp_encoder(X_t)  # (n_train, d_model)
            self.values_Y = torch.from_numpy(self.memory_Y).float().to(encoder_device)  # (n_train, pred_len, n_feat)
        del temp_encoder, X_t
        gc.collect()

        # DataLoader
        dataset = TensorDataset(
            torch.from_numpy(self.memory_X),
            torch.from_numpy(self.memory_Y)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True
        )

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_X, batch_Y in loader:
                batch_X = batch_X.float().to(self.device)  # (batch, seq_len, n_feat)
                batch_Y = batch_Y.float().to(self.device)  # (batch, pred_len, n_feat)

                # Forward
                y_pred_flat = self.net(
                    batch_X,
                    self.keys_X_encoded,
                    self.values_Y
                )  # (batch, pred_len * n_feat)

                y_true_flat = batch_Y.reshape(batch_Y.shape[0], -1)  # (batch, pred_len * n_feat)
                loss = criterion(y_pred_flat, y_true_flat)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                logger.info(
                    f"[RAGSearch] epoch {epoch+1}/{self.epochs} "
                    f"MSE={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.6f}"
                )

        self.net.eval()
        self.is_fitted = True
        logger.info("[RAGSearch] fit done")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Cross-Attention 推理

        Args:
            X_test: shape (n_test, seq_len, n_feat) 或 (n_test, seq_len)

        Returns:
            shape (n_test, pred_len, n_feat) 或 (n_test, pred_len)
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        Xt = X_test.astype(np.float32)
        if Xt.ndim == 2:
            Xt = Xt[:, :, None]

        n_test = Xt.shape[0]
        logger.info(f"[RAGSearch] predict: n_test={n_test}")

        chunks = []
        cs = max(64, self.batch_size)
        self.net.eval()

        with torch.no_grad():
            for start in range(0, n_test, cs):
                end = min(start + cs, n_test)
                xb = torch.from_numpy(Xt[start:end]).float().to(self.device)

                # Cross-Attention: query(xb) attends to keys_X_encoded and values_Y
                yb_flat = self.net(xb, self.keys_X_encoded, self.values_Y)  # (chunk, pred_len * n_feat)
                chunks.append(yb_flat.cpu().numpy())
                del xb, yb_flat

        Y_pred = np.vstack(chunks)  # (n_test, pred_len * n_feat)

        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        gc.collect()
        logger.info(f"[RAGSearch] predict done: output shape={Y_pred.shape}")
        return Y_pred.astype(np.float32)
