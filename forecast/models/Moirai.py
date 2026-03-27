import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """Moirai-2.0 foundation model wrapper for zero-shot forecasting.
    Uses Salesforce/moirai-2.0-R-small via uni2ts library.
    Channel-independent: processes each variate separately.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'moirai_model', 'Salesforce/moirai-2.0-R-small')
        self.device_str = getattr(configs, 'device', 'cpu')

        self._model = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        self._model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_name),
            prediction_length=self.pred_len,
            context_length=self.seq_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self.device_str)
        self._model.eval()
        self._loaded = True

    def forward(self, x):
        raise NotImplementedError("Moirai is inference-only. Use predict() instead.")

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
            # past_target: list of [seq_len, 1]
            past_target = [x_np[b, :, c:c+1] for b in range(B)]
            pred = self._model.predict(past_target=past_target)
            # pred: [B, n_quantiles, pred_len] -> take median
            median_idx = pred.shape[1] // 2
            predictions[:, :, c] = pred[:, median_idx, :]

        return predictions
