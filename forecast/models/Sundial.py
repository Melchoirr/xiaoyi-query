import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """Sundial foundation model wrapper for zero-shot forecasting.
    Uses thuml/sundial-base-128m from HuggingFace.
    Channel-independent: processes each variate separately.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'sundial_model', 'thuml/sundial-base-128m')
        self.device = getattr(configs, 'device', 'cpu')

        self._pipeline = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from transformers import pipeline
        self._pipeline = pipeline(
            "time-series-forecasting",
            model=self.model_name,
            device=self.device,
        )
        self._loaded = True

    def forward(self, x):
        raise NotImplementedError("Sundial is inference-only. Use predict() instead.")

    @torch.no_grad()
    def predict(self, x):
        """
        x: [B, L, C] numpy array or torch tensor (already scaled)
        Returns: [B, pred_len, C] numpy array
        """
        self._load_model()

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        B, L, C = x.shape
        predictions = np.zeros((B, self.pred_len, C))

        for c in range(C):
            for b in range(B):
                series = x[b, :, c].tolist()
                output = self._pipeline(
                    series,
                    prediction_length=self.pred_len,
                )
                # pipeline returns list of dicts with 'mean' key
                if isinstance(output, list) and len(output) > 0:
                    if isinstance(output[0], dict) and 'mean' in output[0]:
                        pred = np.array(output[0]['mean'])[:self.pred_len]
                    else:
                        pred = np.array(output[0])[:self.pred_len]
                else:
                    pred = np.zeros(self.pred_len)
                predictions[b, :len(pred), c] = pred

        return predictions
