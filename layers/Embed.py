import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, L, D]
        return x + self.pe[:, :x.size(1), :]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding=0, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C] -> patch per channel
        B, L, C = x.shape
        # channel-independent: reshape to [B*C, L]
        x = x.permute(0, 2, 1).reshape(B * C, L)  # [B*C, L]
        x = self.padding_patch_layer(x.unsqueeze(1))  # [B*C, 1, L+pad]
        x = x.squeeze(1)  # [B*C, L+pad]
        # unfold into patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*C, num_patches, patch_len]
        x = self.value_embedding(x)  # [B*C, num_patches, d_model]
        x = self.position_encoding(x)
        x = self.dropout(x)
        return x, B, C  # [B*C, num_patches, d_model]
