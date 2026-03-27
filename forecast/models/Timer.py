import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """Timer-XL foundation model wrapper for zero-shot forecasting.
    Uses thuml/timer-base-84m. Channel-independent via batch flattening.
    """
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
            self.model_name,
            trust_remote_code=True,
            torch_dtype='auto',
        ).to(self.device_str)
        self._model.eval()
        self._loaded = True

    def forward(self, x):
        raise NotImplementedError("Timer is inference-only. Use predict() instead.")

    @torch.no_grad()
    def predict(self, x):
        """
        x: [B, L, C] torch tensor (already scaled)
        Returns: [B, pred_len, C] numpy array
        """
        self._load_model()

        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        B, L, C = x_np.shape
        # Flatten channels: [B, L, C] -> [B*C, L]
        flat = x_np.transpose(0, 2, 1).reshape(B * C, L)
        batch_tensor = torch.from_numpy(flat).to(
            device=self.device_str, dtype=self._model.dtype
        )

        outputs = self._model.generate(
            batch_tensor,
            max_new_tokens=self.pred_len,
            do_sample=False,
        )
        result = outputs[:, -self.pred_len:].float().cpu().numpy()  # [B*C, pred_len]
        return result.reshape(B, C, self.pred_len).transpose(0, 2, 1)  # [B, pred_len, C]
