import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Normalizes input per-instance per-channel, and denormalizes output
    using the stored statistics. Supports optional learnable affine.

    Usage:
        revin = RevIN(num_features=7, affine=True)
        x_norm = revin.normalize(x)    # [B, L, C]
        ...  # model forward
        x_out = revin.denormalize(out)  # [B, pred_len, C]
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):
        # x: [B, L, C]
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x = (x - self._mean) / self._stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        # x: [B, pred_len, C]
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self._stdev + self._mean
        return x
