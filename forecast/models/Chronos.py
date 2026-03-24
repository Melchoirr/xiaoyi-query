import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """Chronos foundation model wrapper for zero-shot forecasting.
    Supports both Chronos-Bolt and Chronos-T5 variants.
    Channel-independent: processes each variate separately.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'chronos_model', 'amazon/chronos-bolt-small')
        self.device_str = getattr(configs, 'device', 'cpu')

        self._pipeline = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from chronos import BaseChronosPipeline
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device_str,
            torch_dtype=torch.float32,
        )
        self._loaded = True

    def forward(self, x):
        raise NotImplementedError("Chronos is inference-only. Use predict() instead.")

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
            # prepare batch of univariate series for this channel
            context = [torch.tensor(x_np[b, :, c], dtype=torch.float32) for b in range(B)]

            quantiles, mean = self._pipeline.predict_quantiles(
                context,
                prediction_length=self.pred_len,
                quantile_levels=[0.5],
            )
            # mean: [B, pred_len]
            mean_np = mean.numpy() if isinstance(mean, torch.Tensor) else np.array(mean)
            predictions[:, :, c] = mean_np[:, :self.pred_len]

        return predictions
