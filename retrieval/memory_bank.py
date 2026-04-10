from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MemoryItem:
    history: np.ndarray
    future: np.ndarray
    timestamp: Optional[np.ndarray] = None
    meta: Optional[dict] = None


class MemoryBank:
    def __init__(self) -> None:
        self.histories: Optional[np.ndarray] = None
        self.futures_raw: Optional[np.ndarray] = None
        self.futures_norm: Optional[np.ndarray] = None
        self.futures_delta: Optional[np.ndarray] = None

        self.hist_mean: Optional[np.ndarray] = None
        self.hist_std: Optional[np.ndarray] = None
        self.hist_last: Optional[np.ndarray] = None
        self.hist_slope: Optional[np.ndarray] = None
        self.phase: Optional[np.ndarray] = None
        self.meta: Optional[dict] = None

    def build(self, histories: np.ndarray, futures: np.ndarray, meta: Optional[dict] = None) -> "MemoryBank":
        if histories.ndim != 3:
            raise ValueError("histories must be [N, seq_len, C]")
        if futures.ndim != 3:
            raise ValueError("futures must be [N, pred_len, C]")
        if histories.shape[0] != futures.shape[0]:
            raise ValueError("histories/futures sample size mismatch")
        if histories.shape[2] != futures.shape[2]:
            raise ValueError("histories/futures channel mismatch")

        self.histories = histories.astype(np.float32, copy=False)
        self.futures_raw = futures.astype(np.float32, copy=False)

        self.hist_mean = self.histories.mean(axis=1).astype(np.float32)
        self.hist_std = np.clip(self.histories.std(axis=1), 1e-6, None).astype(np.float32)
        self.hist_last = self.histories[:, -1, :].astype(np.float32)

        t = np.arange(self.histories.shape[1], dtype=np.float32)
        t = (t - t.mean()) / np.clip(t.std(), 1e-6, None)
        centered = self.histories - self.hist_mean[:, None, :]
        self.hist_slope = (centered * t[None, :, None]).mean(axis=1).astype(np.float32)

        self.futures_norm = ((self.futures_raw - self.hist_mean[:, None, :]) / self.hist_std[:, None, :]).astype(np.float32)
        self.futures_delta = (self.futures_raw - self.hist_last[:, None, :]).astype(np.float32)

        self.meta = meta or {}
        phase = self.meta.get("phase")
        if phase is not None:
            self.phase = np.asarray(phase, dtype=np.float32)
        else:
            self.phase = None
        return self

    @property
    def size(self) -> int:
        return 0 if self.histories is None else int(self.histories.shape[0])

    def is_ready(self) -> bool:
        return self.histories is not None and self.futures_raw is not None
