import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'timer_model', 'thuml/timer-base-84m')
        self.device_str = getattr(configs, 'device', 'cpu')
        self._model = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype='auto',
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
        batch_tensor = torch.from_numpy(flat).to(device=self.device_str, dtype=self._model.dtype)

        outputs = self._model.generate(batch_tensor, max_new_tokens=self.pred_len)
        result = outputs[:, -self.pred_len:].float().cpu().numpy()
        return result.reshape(B, C, self.pred_len).transpose(0, 2, 1)
