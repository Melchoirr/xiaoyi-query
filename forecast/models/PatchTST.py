import torch
import torch.nn as nn
from forecast.layers.Embed import PatchEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout

        # calculate number of patches
        self.padding = self.stride
        self.patch_num = int((self.seq_len - self.patch_len + self.padding) / self.stride + 1)

        # patch embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride,
            padding=self.padding, dropout=self.dropout
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

        # prediction head
        self.head = nn.Linear(self.patch_num * self.d_model, self.pred_len)
        self.head_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape

        # patch embedding (channel-independent)
        x_patch, _, _ = self.patch_embedding(x)  # [B*C, num_patches, d_model]

        # transformer encoder
        x_patch = self.encoder(x_patch)  # [B*C, num_patches, d_model]

        # flatten and predict
        x_patch = x_patch.reshape(B * C, -1)  # [B*C, num_patches * d_model]
        x_patch = self.head_dropout(x_patch)
        x_out = self.head(x_patch)  # [B*C, pred_len]

        # reshape back
        x_out = x_out.reshape(B, C, self.pred_len)  # [B, C, pred_len]
        x_out = x_out.permute(0, 2, 1)  # [B, pred_len, C]

        return x_out
