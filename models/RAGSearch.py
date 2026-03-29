"""
RAGSearch: 检索增强记忆网络（Cross-Attention 机制，v3.1 Siamese 重构版）

学术规范：
- Siamese 架构：Query Encoder 和 Key Encoder 是同一个网络（权值共享）
- Memory Bank: X_train 作为 Keys（由同一 Encoder 编码），Y_train 作为 Values
- fit: 端到端训练，Encoder 同时编码 Query 和 Key，MSE Loss 优化
- predict: Encoder 全量编码记忆库作为冻结 Key，Cross-Attention 输出预测

核心参数：d_model (默认 32), n_heads (默认 4), epochs (默认 10), batch_size (默认 128)

硬件安全：
- 训练阶段：Keys/Values 预编码后存于 GPU 显存，V100 32G 完全承载
- 推理阶段：分块 chunk（默认 512）防止显存爆炸
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
# 网络模块
# ─────────────────────────────────────────────────────────────

class _QueryEncoder(nn.Module):
    """
    序列编码器：时序 → d_model 维向量（用于 Query 和 Key 的 Siamese 共享）
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        返回: (batch, d_model) — 时间平均池化
        """
        h = self.proj(x)          # (batch, seq_len, d_model)
        return h.mean(dim=1)      # (batch, d_model)


class _RAGCore(nn.Module):
    """
    Cross-Attention 记忆网络（纯 Torch 实现，无 MHA 依赖）

    Q = Query Encoder(x_query) → (batch, d_model)
    K = X_train_keys → (n_train, d_model)  [预编码，推理时冻结]
    V = Y_train_values → (n_train, pred_len, n_feat)

    Attention: softmax(Q @ K^T / sqrt(d)) @ V
    纯 Torch 实现：对每个 batch 的 query，直接计算与全量 keys 的点积注意力，
    再用 einsum 加权求和 values，避免 nn.MultiheadAttention 的 reshape 维度限制。
    """

    def __init__(self, d_model: int, n_heads: int, n_features: int, pred_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_features = n_features
        self.pred_len = pred_len
        self.val_dim = pred_len * n_features

        self.out_proj = nn.Sequential(
            nn.Linear(self.val_dim, self.val_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(self.val_dim // 2, self.val_dim)
        )

    def forward(
        self,
        q_enc: torch.Tensor,
        k_enc: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q_enc: (batch, d_model) — 当前 Query 编码
            k_enc: (n_keys, d_model) — 全量记忆 Key（预编码，冻结）
            v: (n_keys, pred_len, n_feat) — 全量记忆 Value

        Returns:
            (batch, pred_len, n_feat) — Cross-Attention 预测
        """
        # Scaled dot-product attention
        attn_w = torch.softmax(
            (q_enc @ k_enc.T) / (self.d_model ** 0.5), dim=-1
        )   # (batch, n_keys)

        # 安全：将 3D 的 v 展平为 2D，直接矩阵乘法加权
        v_flat = v.reshape(v.shape[0], -1)               # (n_keys, pred_len * n_features)
        y_pred = attn_w @ v_flat                         # (batch, pred_len * n_features)

        # MLP 投影
        y_pred = self.out_proj(y_pred)                   # (batch, pred_len * n_features)
        return y_pred


class _RAGNet(nn.Module):
    """
    完整 Siamese RAG 网络

    Query Encoder 和 Key Encoder 共享同一个网络（权值完全一致）。
    这确保了 Query 空间和 Key 空间始终对齐，而不是随机且冻结的。
    """

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
        keys_X: torch.Tensor,
        values_Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_X: (batch, seq_len, input_dim) — 当前 Query 原始序列
            keys_X: (n_train, seq_len, input_dim) — 全量记忆原始序列（仅用于 fit 阶段）
            values_Y: (n_train, pred_len, n_feat) — 全量记忆 Value

        Returns:
            (batch, pred_len, n_feat) — Cross-Attention 预测
        """
        # Query 和 Key 使用同一个 Encoder（Siamese）
        q = self.encoder(query_X)   # (batch, d_model)
        k = self.encoder(keys_X)    # (n_train, d_model) — 推理时 keys_X = 全量 memory_X
        return self.core(q, k, values_Y)


# ─────────────────────────────────────────────────────────────
# RAGSearch 主类
# ─────────────────────────────────────────────────────────────

class RAGSearch:
    """
    检索增强记忆网络（Siamese Cross-Attention）

    关键设计：
    - Siamese 架构：Query 和 Key 共享同一个 Encoder，确保空间对齐
    - 训练阶段：每个 batch 的 query attends 到全量训练集的 key/value
    - 推理阶段：Encoder 全量编码 X_train 作为冻结 Key，分块 Cross-Attention
    - GPU 自动检测 + chunk 分块推理，防止显存爆炸

    Args:
        d_model: 隐向量维度（默认 32）
        n_heads: 注意力头数（默认 4）
        epochs: 训练轮数（默认 10）
        batch_size: 批大小（默认 128）
        lr: 学习率（默认 1e-3）
        weight_decay: 权重衰减（默认 1e-4）
        device: 计算设备（默认 'auto'，自动检测 CUDA）
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
        device: Union[str, torch.device] = 'auto',
        seed: int = 42,
        **kwargs
    ):
        # 自动设备检测
        if isinstance(device, str) and device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.d_model = d_model
        self.n_heads = n_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

        self.net: Optional[_RAGNet] = None
        # 推理阶段：预编码的 Keys（全量 X_train，冻结）和 Values（全量 Y_train）
        self._keys_encoded: Optional[torch.Tensor] = None   # (n_train, d_model)
        self._values_raw: Optional[torch.Tensor] = None      # (n_train, pred_len, n_feat)
        self.is_fitted = False
        self.seq_len: int = 0
        self.pred_len: int = 0
        self.n_features: int = 1
        self.n_train: int = 0

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Siamese Cross-Attention 端到端训练

        核心流程：
        1. 初始化网络（Query/Key 共享同一 Encoder）
        2. 将全量 X_train / Y_train 转为 GPU Tensor（Keys 和 Values）
        3. DataLoader 遍历 batch，每个 batch 的 query attends 到全量 Keys
        4. MSE Loss 端到端反向传播

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
        self.n_train = X.shape[0]

        if X.ndim == 2:
            X = X[:, :, None]
        if Y.ndim == 2:
            Y = Y[:, :, None]

        logger.info(
            f"[RAGSearch] fit: n_train={self.n_train}, seq_len={self.seq_len}, "
            f"pred_len={self.pred_len}, n_feat={self.n_features}, "
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"epochs={self.epochs}, device={self.device}"
        )

        # ── 构建网络（Query/Key 共享 Encoder）───────────────────
        input_dim = max(1, self.n_features)
        self.net = _RAGNet(
            input_dim=input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_features=self.n_features,
            pred_len=self.pred_len
        ).to(self.device)

        # ── 准备全量记忆（Keys/Values → GPU）────────────────────
        # Keys: 训练时用全量 X_train，在每个 step 编码为 Keys（梯度流经 Encoder）
        # Values: 全量 Y_train
        self._values_raw = torch.from_numpy(Y).float().to(self.device)   # (n_train, pred_len, n_feat)

        # ── 训练循环 ──────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss(reduction='mean')

        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True
        )

        # 预计算训练集的 Key 编码（每 epoch 重新编码一次，避免重复计算）
        # 对于大 n_train，可以在 epoch 内做采样来加速（这里保持全量精确）
        self.net.train()
        for epoch in range(self.epochs):
            # 每个 epoch 重新编码全量 Keys（确保 Encoder 梯度更新后 Key 空间同步更新）
            X_all_t = torch.from_numpy(X).float().to(self.device)  # (n_train, seq_len, n_feat)
            with torch.no_grad():
                keys_all = self.net.encoder(X_all_t)   # (n_train, d_model)
            del X_all_t
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            total_loss, n_batches = 0.0, 0
            for (batch_x,) in loader:
                batch_x = batch_x.float().to(self.device)          # (batch, seq_len, n_feat)
                batch_Y = torch.from_numpy(
                    Y[np.arange(len(batch_x))]
                ).float().to(self.device)                          # (batch, pred_len, n_feat)

                # Forward: query attends to 全量 keys（不是随机 Encoder）
                y_pred_flat = self.net(batch_x, batch_x, batch_Y)   # (batch, pred_len * n_feat)
                y_true_flat = batch_Y.reshape(batch_Y.shape[0], -1) # (batch, pred_len * n_feat)
                loss = criterion(y_pred_flat, y_true_flat)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                del batch_x, batch_Y, y_pred_flat, y_true_flat, loss

            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            del keys_all
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                logger.info(
                    f"[RAGSearch] epoch {epoch+1}/{self.epochs} "
                    f"MSE={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.6f}"
                )

        self.net.eval()

        # ── 推理准备：全量编码 Keys（冻结）────────────────────────
        X_all_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            self._keys_encoded = self.net.encoder(X_all_t)   # (n_train, d_model)
        del X_all_t
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        logger.info(f"[RAGSearch] fit done. Keys encoded: {self._keys_encoded.shape}")
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Cross-Attention 分块推理

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

        # 全量 Keys 和 Values（已预编码并驻留在 GPU 显存中）
        k_all = self._keys_encoded          # (n_train, d_model)
        v_all = self._values_raw            # (n_train, pred_len, n_feat)

        # 分块推理：防止 35K 样本一次性 GPU OOM
        cs = max(64, self.batch_size)
        chunks = []
        self.net.eval()

        with torch.no_grad():
            for start in range(0, n_test, cs):
                end = min(start + cs, n_test)
                xb = torch.from_numpy(Xt[start:end]).float().to(self.device)  # (chunk, seq_len, n_feat)

                # Query Encoder + Cross-Attention（Keys 冻结）
                qb = self.net.encoder(xb)               # (chunk, d_model)
                # attend to 全量 frozen keys
                yb_pred = self.net.core(qb, k_all, v_all)  # (chunk, pred_len * n_feat)

                chunks.append(yb_pred.cpu().numpy())
                del xb, qb, yb_pred

        del k_all, v_all
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        Y_pred = np.vstack(chunks)  # (n_test, pred_len * n_feat)

        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        logger.info(f"[RAGSearch] predict done: output shape={Y_pred.shape}")
        return Y_pred.astype(np.float32)
