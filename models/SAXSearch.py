"""
SAXSearch: 符号聚合近似检索基线
- SAX 符号化向量化（numpy 广播）
- predict 精确匹配阶段用 np.unique + 批量赋值，避免逐样本 dict
- 可选 GPU：实例归一化在 torch 上完成（减轻大数据矩阵在 CPU 上的开销）
"""

import gc
import logging
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import torch
from scipy.stats import norm

logger = logging.getLogger(__name__)


class SAXSearch:
    DTYPE = np.float32

    def __init__(
        self,
        word_size: int = 8,
        alphabet_size: int = 8,
        epsilon_threshold: float = 1.0,
        fallback_strategy: str = 'global_mean',
        random_state: Optional[int] = 42,
        seq_len: int = 0,
        pred_len: int = 0,
        n_features: int = 1,
        device: Union[str, torch.device] = 'cpu',
        **kwargs
    ):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.epsilon_threshold = epsilon_threshold
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]
        self._alphabet_np = np.array(self.alphabet, dtype='U1')
        self.breakpoints = self._compute_breakpoints()

        self.sax_dict: Dict[str, Tuple[np.ndarray, int]] = {}
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted = False
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len

    def _compute_breakpoints(self) -> np.ndarray:
        breakpoints = norm.ppf(np.linspace(
            1.0 / self.alphabet_size,
            (self.alphabet_size - 1) / self.alphabet_size,
            self.alphabet_size - 1
        ))
        return breakpoints.astype(self.DTYPE)

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.device.type == 'cuda':
            with torch.no_grad():
                t = torch.from_numpy(X.astype(np.float32)).to(self.device)
                X_mean = t.mean(dim=1, keepdim=True)
                X_std = t.std(dim=1, keepdim=True).clamp(min=1e-8)
                X_norm = ((t - X_mean) / X_std).cpu().numpy().astype(self.DTYPE)
            # 返回占位 mean/std 保持接口一致
            return X_norm, np.zeros((X.shape[0], 1), dtype=self.DTYPE), np.ones((X.shape[0], 1), dtype=self.DTYPE)

        if X.ndim == 3:
            X_mean = np.mean(X, axis=1, keepdims=True)
            X_std = np.std(X, axis=1, keepdims=True)
        else:
            X_mean = np.mean(X, axis=1, keepdims=True)
            X_std = np.std(X, axis=1, keepdims=True)
        X_std = np.clip(X_std, 1e-8, None)
        X_norm = (X - X_mean) / X_std
        return X_norm, X_mean, X_std

    def _paa_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len = X.shape
        segment_size = seq_len / self.word_size
        paa = np.zeros((n_samples, self.word_size), dtype=self.DTYPE)
        for i in range(self.word_size):
            start = int(i * segment_size)
            end = int((i + 1) * segment_size)
            if i == self.word_size - 1:
                end = seq_len
            paa[:, i] = np.mean(X[:, start:end], axis=1)
        return paa

    def _sax_strings_from_paa(self, paa: np.ndarray) -> np.ndarray:
        """
        向量化：paa (n, word_size) -> SAX 字符串数组 (n,) object dtype
        """
        bp = self.breakpoints.astype(np.float64)
        if bp.size == 0:
            idx = np.zeros(paa.shape, dtype=np.int64)
        else:
            idx = (paa[..., None] > bp.reshape(1, 1, -1)).sum(axis=-1).astype(np.int64)
        idx = np.clip(idx, 0, self.alphabet_size - 1)
        chars = self._alphabet_np[idx]
        n = paa.shape[0]
        out = np.empty(n, dtype=object)
        for i in range(n):
            out[i] = ''.join(chars[i].tolist())
        return out

    def _compute_edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._compute_edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        n_samples = X_train.shape[0]

        if X_train.ndim == 3:
            self.seq_len = X_train.shape[1]
            self.n_features = X_train.shape[2]
            X_flat = X_train.reshape(n_samples, -1).astype(self.DTYPE)
        else:
            self.seq_len = X_train.shape[1]
            self.n_features = 1
            X_flat = X_train.reshape(n_samples, -1).astype(self.DTYPE)

        if Y_train.ndim == 3:
            self.pred_len = Y_train.shape[1]
            Y_original = Y_train.astype(self.DTYPE)
        else:
            self.pred_len = Y_train.shape[1]
            self.n_features = 1
            Y_original = Y_train.astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)
        paa = self._paa_transform(X_norm)
        sax_arr = self._sax_strings_from_paa(paa)

        sum_cache: Dict[str, np.ndarray] = {}
        count_cache: Dict[str, int] = {}

        for i in range(n_samples):
            sax_str = sax_arr[i]
            if sax_str not in sum_cache:
                sum_cache[sax_str] = np.zeros(
                    (self.pred_len, self.n_features), dtype=self.DTYPE)
                count_cache[sax_str] = 0
            sum_cache[sax_str] += Y_original[i]
            count_cache[sax_str] += 1

        for sax_str in sum_cache:
            cnt = count_cache[sax_str]
            mean_Y = (sum_cache[sax_str] / cnt).astype(self.DTYPE)
            self.sax_dict[sax_str] = (mean_Y, cnt)

        self.global_Y_mean = np.mean(Y_original, axis=0).astype(self.DTYPE)
        del Y_original, X_flat, X_norm, paa, sax_arr, sum_cache, count_cache
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.is_fitted = True

        logger.info(f"[SAXSearch] fitted: {len(self.sax_dict)} unique strings, device={self.device}")
        return self

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]
        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        X_norm, _, _ = self._instance_normalize(X_flat)
        paa = self._paa_transform(X_norm)
        test_sax = self._sax_strings_from_paa(paa)

        Y_pred = np.zeros((n_test, self.pred_len, self.n_features), dtype=self.DTYPE)
        edit_threshold = int(self.epsilon_threshold * self.word_size)
        all_sax_keys = list(self.sax_dict.keys())

        matched = np.zeros(n_test, dtype=bool)

        # ── 精确匹配：按唯一字符串批量写入 ──
        uniq, inv = np.unique(test_sax, return_inverse=True)
        for j, s in enumerate(uniq):
            if s not in self.sax_dict:
                continue
            mean_Y, _ = self.sax_dict[s]
            idx = np.nonzero(inv == j)[0]
            Y_pred[idx] = mean_Y
            matched[idx] = True

        matched_count = int(matched.sum())
        fallback_count = 0

        # ── 未命中：模糊匹配或 fallback ──
        todo = np.nonzero(~matched)[0]
        if len(todo) > 0 and edit_threshold > 0 and len(all_sax_keys) > 0:
            for i in todo:
                sax_str = test_sax[i]
                best_key = None
                best_dist = float('inf')
                for stored_str in all_sax_keys:
                    dist = self._compute_edit_distance(sax_str, stored_str)
                    if dist < best_dist and dist <= edit_threshold:
                        best_dist = dist
                        best_key = stored_str
                if best_key is not None:
                    Y_pred[i] = self.sax_dict[best_key][0]
                    matched[i] = True
                    matched_count += 1
                else:
                    fallback_count += 1
                    if self.fallback_strategy == 'global_mean':
                        Y_pred[i] = self.global_Y_mean
                    else:
                        last_vals = X_flat[i].reshape(self.pred_len, self.n_features)
                        Y_pred[i] = np.full(
                            (self.pred_len, self.n_features),
                            float(np.mean(last_vals[-1])),
                            dtype=self.DTYPE,
                        )
        elif len(todo) > 0:
            for i in todo:
                fallback_count += 1
                if self.fallback_strategy == 'global_mean':
                    Y_pred[i] = self.global_Y_mean
                else:
                    last_vals = X_flat[i].reshape(self.pred_len, self.n_features)
                    Y_pred[i] = np.full(
                        (self.pred_len, self.n_features),
                        float(np.mean(last_vals[-1])),
                        dtype=self.DTYPE,
                    )

        if fallback_count > 0:
            logger.warning(f"[SAXSearch] fallback {fallback_count}/{n_test}")
        logger.info(f"[SAXSearch] predict done {Y_pred.shape}, matched={matched_count}, device={self.device}")

        if self.n_features == 1:
            Y_pred = Y_pred.squeeze(-1)

        del X_flat, X_norm, paa, test_sax
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return Y_pred

    def get_params(self) -> dict:
        return {
            'model_type': 'SAXSearch',
            'word_size': self.word_size,
            'alphabet_size': self.alphabet_size,
            'epsilon_threshold': self.epsilon_threshold,
            'device': str(self.device),
            'n_unique_strings': len(self.sax_dict) if self.is_fitted else 0,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
        }

    def get_sax_stats(self) -> dict:
        if not self.is_fitted:
            return {}
        bucket_counts = [cnt for _, cnt in self.sax_dict.values()]
        total_samples = sum(bucket_counts)
        return {
            'n_unique_strings': len(self.sax_dict),
            'total_samples': total_samples,
            'min_bucket': min(bucket_counts) if bucket_counts else 0,
            'max_bucket': max(bucket_counts) if bucket_counts else 0,
            'avg_bucket': float(np.mean(bucket_counts)) if bucket_counts else 0,
        }

    def __repr__(self):
        return (
            f"SAXSearch(word_size={self.word_size}, alphabet_size={self.alphabet_size}, "
            f"device={self.device})"
        )
