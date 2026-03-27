import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'chronos_model', 'amazon/chronos-2')
        self.device_str = getattr(configs, 'device', 'cpu')
        self._pipeline = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from chronos import Chronos2Pipeline
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name,
            device_map=self.device_str,
            torch_dtype=torch.float32,
        )
        self._loaded = True

    def forward(self, x):
        raise NotImplementedError("Use predict()")

    @torch.no_grad()
    def predict(self, x):
        self._load_model()
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        B, L, C = x_np.shape
        # Chronos-2 原生多变量: [B, C, L]
        batch_tensor = torch.from_numpy(x_np.transpose(0, 2, 1))

        predictions = self._pipeline.predict(batch_tensor, prediction_length=self.pred_len)
        # list of [C, n_quantiles, pred_len] tensors
        stacked = torch.stack(predictions)  # [B, C, n_quantiles, pred_len]
        n_quantiles = stacked.shape[2]
        median_idx = n_quantiles // 2
        result = stacked[:, :, median_idx, :].cpu().numpy()  # [B, C, pred_len]
        return result.transpose(0, 2, 1)  # [B, pred_len, C]
