"""
PatternSearch: 基于记忆检索的时序预测基线模型
使用 torch.cdist + topk 在 GPU/CPU 上批量近邻检索（替代 sklearn KD-Tree 逐批查询）
"""

import gc
import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PatternSearch:
    """
    将历史 X 展平为记忆库，预测时对测试 X 与全库做欧氏距离矩阵，
    用 torch.topk 取最近 k 条并对对应 Y 做（可选）逆距离加权平均。
    """

    DTYPE = np.float32

    def __init__(
        self,
        top_k: int = 5,
        weighted: bool = True,
        algorithm: str = 'kd_tree',  # 保留兼容，torch 路径下忽略
        seq_len: int = 0,
        pred_len: int = 0,
        n_features: int = 1,
        device: Union[str, torch.device] = 'cpu',
        predict_chunk_size: int = 2048,
        **kwargs
    ):
        self.k = top_k
        self.weighted = weighted
        self.algorithm = algorithm
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.predict_chunk_size = max(256, int(predict_chunk_size))

        self.memory_X: Optional[np.ndarray] = None
        self.memory_Y: Optional[np.ndarray] = None
        self._mem_X_t: Optional[torch.Tensor] = None
        self._mem_Y_t: Optional[torch.Tensor] = None
        self.is_fitted = False
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        # ── 严格捕获真实维度，禁止 n_features=1 残留 ──────────────────
        self.seq_len = X_train.shape[1]
        self.pred_len = Y_train.shape[1]
        # Y 永远决定 n_features：3D -> Y.shape[-1]，2D -> X.shape[-1]（展平前）
        if Y_train.ndim == 3:
            self.n_features = Y_train.shape[-1]
        else:  # 2D: Y shape=(n, pred_len), n_features 必须从 X 推导
            self.n_features = X_train.shape[-1] if X_train.ndim == 3 else 1

        n_samples = X_train.shape[0]
        self.memory_X = X_train.reshape(n_samples, -1).astype(self.DTYPE)
        self.memory_Y = Y_train.reshape(n_samples, -1).astype(self.DTYPE)

        self._mem_X_t = torch.from_numpy(self.memory_X).to(
            self.device, dtype=torch.float32
        )
        self._mem_Y_t = torch.from_numpy(self.memory_Y).to(
            self.device, dtype=torch.float32
        )
        self.is_fitted = True

        logger.info(
            f"[PatternSearch] fit device={self.device}, "
            f"memory_X={self.memory_X.shape}, memory_Y={self.memory_Y.shape}"
        )
        return self

    def predict(self, X_test: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        if not self.is_fitted or self._mem_X_t is None or self._mem_Y_t is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        k = top_k if top_k is not None else self.k
        k = min(k, self._mem_X_t.shape[0])

        if X_test.ndim == 3:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        mem_X = self._mem_X_t
        mem_Y = self._mem_Y_t
        y_dim = mem_Y.shape[1]

        chunks = []
        cs = self.predict_chunk_size
        logger.info(
            f"[PatternSearch] predict n_test={n_test}, k={k}, chunk={cs}, device={self.device}"
        )

        with torch.no_grad():
            for start in range(0, n_test, cs):
                end = min(start + cs, n_test)
                xb = torch.from_numpy(X_flat[start:end]).to(
                    self.device, dtype=torch.float32
                )
                # (chunk, n_train)
                dist = torch.cdist(xb, mem_X, p=2)
                vals, idx = torch.topk(dist, k, largest=False, dim=1)
                neighbor_Y = mem_Y[idx]  # (chunk, k, y_dim)

                if self.weighted:
                    vals = torch.clamp(vals, min=1e-10)
                    w = 1.0 / vals
                    w = w / w.sum(dim=1, keepdim=True)
                    yb = (neighbor_Y * w.unsqueeze(-1)).sum(dim=1)
                else:
                    yb = neighbor_Y.mean(dim=1)

                chunks.append(yb.cpu().numpy().astype(self.DTYPE))
                del xb, dist, vals, idx, neighbor_Y, yb

        Y_pred = np.vstack(chunks)

        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        logger.info(f"[PatternSearch] predict done: {Y_pred.shape}")
        return Y_pred

    def get_neighbors(self, X_query: np.ndarray, top_k: Optional[int] = None):
        if not self.is_fitted or self._mem_X_t is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        k = top_k if top_k is not None else self.k
        k = min(k, self._mem_X_t.shape[0])

        if X_query.ndim == 3:
            n_q = X_query.shape[0]
            X_flat = X_query.reshape(n_q, -1).astype(self.DTYPE)
        else:
            n_q = X_query.shape[0]
            X_flat = X_query.reshape(n_q, -1).astype(self.DTYPE)

        dist_list = []
        idx_list = []
        cs = self.predict_chunk_size
        with torch.no_grad():
            for start in range(0, n_q, cs):
                end = min(start + cs, n_q)
                xb = torch.from_numpy(X_flat[start:end]).to(
                    self.device, dtype=torch.float32
                )
                dist = torch.cdist(xb, self._mem_X_t, p=2)
                vals, idx = torch.topk(dist, k, largest=False, dim=1)
                dist_list.append(vals.cpu().numpy())
                idx_list.append(idx.cpu().numpy())

        return np.vstack(dist_list), np.vstack(idx_list)

    def get_params(self) -> dict:
        return {
            'k': self.k,
            'weighted': self.weighted,
            'device': str(self.device),
            'n_memory': self.memory_X.shape[0] if self.is_fitted and self.memory_X is not None else 0,
            'seq_len': self.seq_len if self.is_fitted else 0,
            'pred_len': self.pred_len if self.is_fitted else 0,
            'n_features': self.n_features if self.is_fitted else 0,
        }

    def __repr__(self):
        return f"PatternSearch(k={self.k}, weighted={self.weighted}, device={self.device})"
