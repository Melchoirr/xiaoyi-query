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

    def get_retrieval_meta(
        self,
        X_test: np.ndarray,
        top_k: Optional[int] = None,
        n_samples: int = 5
    ) -> tuple:
        """
        返回溯源证据：用于绘制"历史匹配溯源图"

        Args:
            X_test: shape (n_test, seq_len, n_feat) 归一化后的测试输入
            top_k: 检索的邻居数量（默认使用 self.k）
            n_samples: 返回前 n_samples 个测试样本的元数据（默认 5）

        Returns:
            Y_pred: 预测结果 shape (n_test, pred_len, n_feat)
            retrieval_meta: dict 包含:
                - topk_histories: (n_samples, k, seq_len, n_feat) 匹配到的历史序列
                - topk_futures: (n_samples, k, pred_len, n_feat) 匹配到的未来序列
                - topk_weights: (n_samples, k) 归一化的注意力权重
        """
        if not self.is_fitted or self._mem_X_t is None or self._mem_Y_t is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        k = top_k if top_k is not None else self.k
        k = min(k, self._mem_X_t.shape[0])

        # 处理输入维度
        if X_test.ndim == 3:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            n_test = X_test.shape[0]
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        # 获取完整预测
        Y_pred = self.predict(X_test, top_k=top_k)

        # 只对前 n_samples 个样本保存溯源证据
        n_samples = min(n_samples, n_test)

        # 原始维度
        mem_X_raw = self.memory_X.reshape(-1, self.seq_len, self.n_features)  # (n_train, seq_len, n_feat)
        mem_Y_raw = self.memory_Y.reshape(-1, self.pred_len, self.n_features)  # (n_train, pred_len, n_feat)

        # 对前 n_samples 个样本检索
        with torch.no_grad():
            xb = torch.from_numpy(X_flat[:n_samples]).to(self.device, dtype=torch.float32)
            dist = torch.cdist(xb, self._mem_X_t, p=2)  # (n_samples, n_train)
            vals, idx = torch.topk(dist, k, largest=False, dim=1)  # (n_samples, k)

            # 转换索引为 numpy
            idx_np = idx.cpu().numpy()  # (n_samples, k)
            vals_np = vals.cpu().numpy()  # (n_samples, k)

            # 计算归一化权重（softmax 或逆距离）
            vals_safe = np.clip(vals_np, 1e-10, None)
            weights = 1.0 / vals_safe
            weights = weights / weights.sum(axis=1, keepdims=True)  # (n_samples, k)

            # 提取匹配的 top_k 历史序列 (n_samples, k, seq_len, n_feat)
            topk_histories = mem_X_raw[idx_np]  # (n_samples, k, seq_len, n_feat)

            # 提取匹配的 top_k 未来序列 (n_samples, k, pred_len, n_feat)
            topk_futures = mem_Y_raw[idx_np]  # (n_samples, k, pred_len, n_feat)

            del xb, dist, vals, idx

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        logger.info(
            f"[PatternSearch] Retrieval meta: {n_samples} samples, k={k}, "
            f"hist_shape={topk_histories.shape}, fut_shape={topk_futures.shape}"
        )

        retrieval_meta = {
            'topk_histories': topk_histories.astype(np.float32),
            'topk_futures': topk_futures.astype(np.float32),
            'topk_weights': weights.astype(np.float32),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'model_name': 'PatternSearch',
        }

        return Y_pred, retrieval_meta
