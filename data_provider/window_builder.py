from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_hist_future_from_batch(batch_x: np.ndarray, batch_y: np.ndarray, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if batch_x.ndim != 3:
        raise ValueError("batch_x must be [B, seq_len, C]")
    if batch_y.ndim != 3:
        raise ValueError("batch_y must be [B, label_len+pred_len, C]")
    future = batch_y[:, -pred_len:, :]
    return batch_x.astype(np.float32), future.astype(np.float32)
