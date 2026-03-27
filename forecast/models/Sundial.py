import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """Sundial foundation model wrapper for zero-shot forecasting.
    Uses thuml/sundial-base-128m from HuggingFace.
    Channel-independent: processes each variate separately.
    generate() auto-applies RevIN, so input does NOT need manual normalization.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'sundial_model', 'thuml/sundial-base-128m')
        self.num_samples = 20

        self._model = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True

    def forward(self, x):
        raise NotImplementedError("Sundial is inference-only. Use predict() instead.")

    @torch.no_grad()
    def predict(self, x):
        """
        x: [B, L, C] torch tensor (already scaled by StandardScaler)
        Returns: [B, pred_len, C] numpy array
        """
        self._load_model()

        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        B, L, C = x_np.shape
        predictions = np.zeros((B, self.pred_len, C))

        for c in range(C):
            # Sundial expects [B, L] float tensor
            seqs = torch.tensor(x_np[:, :, c], dtype=torch.float32)
            # output: [B, num_samples, pred_len]
            output = self._model.generate(
                seqs,
                max_new_tokens=self.pred_len,
                num_samples=self.num_samples,
            )
            # take mean across samples as point forecast
            point_forecast = output.mean(dim=1)  # [B, pred_len]
            predictions[:, :, c] = point_forecast.cpu().numpy()

        return predictions
