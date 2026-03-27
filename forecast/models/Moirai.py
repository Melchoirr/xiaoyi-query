import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """Moirai foundation model wrapper for zero-shot forecasting.
    Uses Salesforce/moirai-1.1-R-base via uni2ts library.
    Channel-independent: processes each variate separately.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'moirai_model', 'Salesforce/moirai-1.1-R-base')
        self.num_samples = 20
        # Map internal freq codes to pandas offset aliases
        freq_map = {'h': 'h', 't': 'min', 's': 's', 'd': 'D', 'b': 'B', 'w': 'W', 'm': 'ME'}
        self.freq = freq_map.get(getattr(configs, 'freq', 'h'), 'h')

        self._predictor = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        module = MoiraiModule.from_pretrained(self.model_name)
        forecast_model = MoiraiForecast(
            module=module,
            prediction_length=self.pred_len,
            context_length=self.seq_len,
            patch_size="auto",
            num_samples=self.num_samples,
        )
        self._predictor = forecast_model.create_predictor(batch_size=32)
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
            from gluonts.dataset.common import ListDataset
            ds = ListDataset(
                [{"start": "2000-01-01 00:00:00", "target": x_np[b, :, c]} for b in range(B)],
                freq=self.freq,
            )
            forecasts = list(self._predictor.predict(ds))
            for b, fc in enumerate(forecasts):
                predictions[b, :, c] = fc.mean

        return predictions
