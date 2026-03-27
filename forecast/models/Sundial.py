import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'sundial_model', 'thuml/sundial-base-128m')
        self.num_samples = 20
        self.device_str = getattr(configs, 'device', 'cpu')
        self._model = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM
        # 照搬 run_sundial.py: 不用 torch_dtype='auto'
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True,
        ).to(self.device_str)
        self._model.eval()
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
        flat = x_np.transpose(0, 2, 1).reshape(B * C, L)

        # 照搬 run_sundial.py: float32 上设备，再转 model.dtype
        inputs_tensor = torch.tensor(flat, dtype=torch.float32).to(self.device_str)
        batch_in = inputs_tensor.to(self._model.dtype)

        forecast_raw = self._model.generate(
            batch_in, max_new_tokens=self.pred_len, num_samples=self.num_samples,
        )

        if forecast_raw.shape[-1] > self.pred_len:
            forecast_slice = forecast_raw[..., -self.pred_len:]
        else:
            forecast_slice = forecast_raw

        if forecast_slice.ndim == 3 and forecast_slice.shape[1] == self.num_samples:
            point_forecast = forecast_slice.mean(dim=1)
        else:
            point_forecast = forecast_slice

        result = point_forecast.float().cpu().numpy()
        return result.reshape(B, C, self.pred_len).transpose(0, 2, 1)
