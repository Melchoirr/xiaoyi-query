import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """TimesFM 2.0 foundation model wrapper for zero-shot forecasting.
    Uses google/timesfm-2.0-500m-pytorch from HuggingFace.
    Channel-independent: processes each variate separately.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_name = getattr(configs, 'timesfm_model', 'google/timesfm-2.0-500m-pytorch')
        self.freq = getattr(configs, 'freq', 'h')

        self._model = None
        self._loaded = False

    def _get_freq_indicator(self):
        # 0=high (second/minute/hourly/daily), 1=medium (weekly/monthly), 2=low (quarterly/yearly)
        freq_map = {'s': 0, 't': 0, 'h': 0, 'd': 0, 'b': 0, 'w': 1, 'm': 2}
        return freq_map.get(self.freq, 0)

    def _load_model(self):
        if self._loaded:
            return
        import timesfm
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=self.pred_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=50,
                model_dims=1280,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_name,
            ),
        )
        self._loaded = True

    def forward(self, x):
        raise NotImplementedError("TimesFM is inference-only. Use predict() instead.")

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
        freq_indicator = self._get_freq_indicator()

        for c in range(C):
            # TimesFM expects list of 1D numpy arrays
            forecast_input = [x_np[b, :, c] for b in range(B)]
            frequency_input = [freq_indicator] * B

            point_forecast, _ = self._model.forecast(
                forecast_input,
                freq=frequency_input,
            )
            # point_forecast: [B, horizon_len]
            point_arr = np.array(point_forecast)
            predictions[:, :, c] = point_arr[:, :self.pred_len]

        return predictions
