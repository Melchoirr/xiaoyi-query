import torch
import torch.nn as nn
import math
from forecast.layers.Embed import PatchEmbedding
from forecast.layers.RevIN import RevIN


class ChannelEncoder(nn.Module):
    """将每个非目标通道的原始序列编码为 d_model 向量"""

    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, C-1, L]
        x = self.linear(x)       # [B, C-1, d_model]
        x = self.dropout(x)
        x = self.norm(x)
        return x


class PrimitiveAssignment(nn.Module):
    """软分配通道表征到原语码本，生成 K 个原语 token"""

    def __init__(self, num_primitives, d_model, temperature=1.0):
        super().__init__()
        self.codebook = nn.Parameter(torch.empty(num_primitives, d_model))
        nn.init.xavier_uniform_(self.codebook)
        self.temperature = temperature
        self.scale = math.sqrt(d_model)

    def forward(self, channel_repr):
        # channel_repr: [B, C-1, d_model]
        # codebook: [K, d_model]
        sim = channel_repr @ self.codebook.T / self.scale   # [B, C-1, K]
        weights = torch.softmax(sim / self.temperature, dim=-1)  # [B, C-1, K]
        # 每个原语 = 按亲和度加权的通道表征混合
        primitive_tokens = weights.permute(0, 2, 1) @ channel_repr  # [B, K, d_model]
        return primitive_tokens, weights


class CrossAttentionLayer(nn.Module):
    """目标通道 attend to 原语 token 的 cross-attention 层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, target_repr, primitive_tokens):
        # Pre-norm cross-attention with residual
        x = self.norm1(target_repr)
        attn_out, _ = self.cross_attn(query=x, key=primitive_tokens, value=primitive_tokens)
        x = target_repr + attn_out

        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.num_primitives = getattr(configs, 'num_primitives', 16)
        self.primitive_temp = getattr(configs, 'primitive_temp', 1.0)
        self.n_cross_layers = getattr(configs, 'n_cross_layers', 1)

        # RevIN
        self.use_revin = getattr(configs, 'revin', False)
        if self.use_revin:
            self.revin = RevIN(configs.enc_in)

        # --- 目标通道路径 (与 PatchTST 相同) ---
        self.padding = self.stride
        self.patch_num = int((self.seq_len - self.patch_len + self.padding) / self.stride + 1)

        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride,
            padding=self.padding, dropout=self.dropout,
        )

        fc_dropout = getattr(configs, 'fc_dropout', self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=fc_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

        # --- 其他通道路径 ---
        self.channel_encoder = ChannelEncoder(self.seq_len, self.d_model, self.dropout)
        self.primitive_assignment = PrimitiveAssignment(
            self.num_primitives, self.d_model, self.primitive_temp
        )

        # --- Cross-attention 融合 ---
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_cross_layers)
        ])

        # --- 预测头 ---
        head_dropout = getattr(configs, 'head_dropout', 0.0)
        self.head = nn.Linear(self.patch_num * self.d_model, self.pred_len)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [B, L, C]
        if self.use_revin:
            x = self.revin.normalize(x)

        B, L, C = x.shape
        outputs = []

        for c in range(C):
            # --- 目标通道 ---
            x_t = x[:, :, c:c + 1]  # [B, L, 1]

            # Patch embedding (CI: 单通道输入)
            x_patch, _, _ = self.patch_embedding(x_t)  # [B*1, N_patches, d_model] = [B, N, d]

            # Self-attention encoder
            target_repr = self.encoder(x_patch)  # [B, N_patches, d_model]

            # --- 其他通道 → 原语 → Cross-attention ---
            if C > 1:
                other_idx = [i for i in range(C) if i != c]
                x_o = x[:, :, other_idx]           # [B, L, C-1]
                x_o = x_o.permute(0, 2, 1)         # [B, C-1, L]

                channel_repr = self.channel_encoder(x_o)  # [B, C-1, d_model]
                primitive_tokens, _ = self.primitive_assignment(channel_repr)  # [B, K, d_model]

                fused = target_repr
                for cross_layer in self.cross_attention_layers:
                    fused = cross_layer(fused, primitive_tokens)
            else:
                fused = target_repr

            # --- 预测 ---
            fused = fused.reshape(B, -1)          # [B, N_patches * d_model]
            fused = self.head_dropout(fused)
            pred_c = self.head(fused)             # [B, pred_len]
            outputs.append(pred_c)

        # Stack all channels
        x_out = torch.stack(outputs, dim=-1)      # [B, pred_len, C]

        if self.use_revin:
            x_out = self.revin.denormalize(x_out)

        return x_out
