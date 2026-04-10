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
        self.futures: Optional[np.ndarray] = None
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
        self.futures = futures.astype(np.float32, copy=False)
        self.meta = meta or {}
        return self

    @property
    def size(self) -> int:
        return 0 if self.histories is None else int(self.histories.shape[0])

    def is_ready(self) -> bool:
        return self.histories is not None and self.futures is not None
