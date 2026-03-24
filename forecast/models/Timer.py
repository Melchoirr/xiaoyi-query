import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """Timer foundation model wrapper for zero-shot forecasting.
    Uses thuml/timer-base-84m from HuggingFace.
    Channel-independent: processes each variate separately.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'timer_model', 'thuml/timer-base-84m')

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
        predictions = np.zeros((B, self.pred_len, C))

        for c in range(C):
            # Timer expects [B, L] float tensor
            seqs = torch.tensor(x_np[:, :, c], dtype=torch.float32)
            output = self._model.generate(seqs, max_new_tokens=self.pred_len)
            # output: [B, pred_len]
            predictions[:, :, c] = output.cpu().numpy()

        return predictions
