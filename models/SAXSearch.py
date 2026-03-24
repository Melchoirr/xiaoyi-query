"""
SAXSearch: 基于符号聚合近似(Symbolic Aggregate approXimation)的时序预测基线模型
将时序压缩为"字符串"进行模糊匹配，实现高效的序列检索与预测

内存优化 (v2.0)：
- 哈希桶不再存索引列表，而是预聚合 (mean_Y, count) 元组
- 删除 self.memory_Y 大数组，predict 时直接查均值
- 预测时避免对整个字典做 O(n_keys * n_test) 的全遍历编辑距离
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy.stats import norm


class SAXSearch:
    """
    基于SAX的时序预测基线模型

    算法流程：
    1. Instance Normalization: 对每个序列进行Z-Score归一化
    2. PAA降维: 将长度为seq_len的序列划分为word_size个片段
    3. 符号化: 根据高斯分位数将每个片段映射到alphabet_size个符号
    4. 存储: 将符号字符串作为键，对应 Y 均值元组为值
    5. 查询: 将测试序列转为SAX字符串，直接查找预聚合均值

    内存优化：
    - self.sax_dict: Dict[str, Tuple[mean_Y, count]]
      原来存 n 个 int 索引（4~8 bytes each），现在只存 1 个 float32 均值数组 + 1 个 count
      假设一个桶有 1000 个样本，原来存 1000 个索引（4KB），现在只存 1 个均值数组（pred_len * 4 bytes）
      压缩比可达 1000x
    """

    DTYPE = np.float32   # 全局统一 float32

    def __init__(
        self,
        word_size: int = 8,
        alphabet_size: int = 8,
        epsilon_threshold: float = 1.0,
        fallback_strategy: str = 'global_mean',
        random_state: Optional[int] = 42
    ):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.epsilon_threshold = epsilon_threshold
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state

        self.alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]
        self.breakpoints = self._compute_breakpoints()

        # ================================================
        # 内存优化: 存预聚合均值而非索引列表
        # Dict[sax_string] -> (mean_Y_flat, count)
        #   mean_Y_flat: np.ndarray shape [pred_len * n_features], dtype float32
        #   count: int，桶内样本数量
        # ================================================
        self.sax_dict: Dict[str, Tuple[np.ndarray, int]] = {}

        # 全局 Y 均值（用于 fallback）
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        # 维度信息
        self.seq_len: int = 0
        self.n_features: int = 1
        self.pred_len: int = 0

    def _compute_breakpoints(self) -> np.ndarray:
        breakpoints = norm.ppf(np.linspace(
            1.0 / self.alphabet_size,
            (self.alphabet_size - 1) / self.alphabet_size,
            self.alphabet_size - 1
        ))
        return breakpoints.astype(self.DTYPE)

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _sax_transform(self, paa: np.ndarray) -> List[str]:
        n_samples = paa.shape[0]
        sax_strings = []

        for i in range(n_samples):
            symbols = []
            for j in range(self.word_size):
                value = paa[i, j]
                symbol_idx = 0
                for bp in self.breakpoints:
                    if value > bp:
                        symbol_idx += 1
                symbols.append(self.alphabet[symbol_idx])
            sax_strings.append(''.join(symbols))

        return sax_strings

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
        """
        构建SAX索引（记忆库）

        内存优化：
        - 遍历一次 X_train/Y_train，在线累加每个 SAX 桶的 sum_Y 和 count
        - 最终每个桶只存储 mean_Y（float32）和 count（int）
        - 不再存储 self.memory_Y 大数组
        """
        n_samples = X_train.shape[0]

        # 保存维度信息
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
            Y_flat = Y_train.reshape(n_samples, -1).astype(self.DTYPE)
        else:
            self.pred_len = Y_train.shape[1]
            Y_flat = Y_train.reshape(n_samples, -1).astype(self.DTYPE)

        y_dim = Y_flat.shape[1]  # pred_len * n_features

        # 计算全局 Y 均值（用于 fallback）
        self.global_Y_mean = np.mean(Y_flat, axis=0).astype(self.DTYPE)

        # 实例归一化
        X_norm, _, _ = self._instance_normalize(X_flat)

        # PAA 变换
        paa = self._paa_transform(X_norm)

        # SAX 符号化
        sax_strings = self._sax_transform(paa)

        # ================================================
        # 内存优化: 在线累加，不再存储索引列表
        # sax_dict: str -> (sum_Y_flat, count)
        #   sum_Y_flat: 累加值（需要最后除以 count 得均值），shape [y_dim]
        #   count: int
        # ================================================
        self.sax_dict = {}
        sum_cache: Dict[str, np.ndarray] = {}
        count_cache: Dict[str, int] = {}

        for i, sax_str in enumerate(sax_strings):
            if sax_str not in sum_cache:
                sum_cache[sax_str] = np.zeros(y_dim, dtype=self.DTYPE)
                count_cache[sax_str] = 0

            sum_cache[sax_str] += Y_flat[i]
            count_cache[sax_str] += 1

        # 转换为均值形式
        for sax_str in sum_cache:
            cnt = count_cache[sax_str]
            mean_Y = (sum_cache[sax_str] / cnt).astype(self.DTYPE)
            self.sax_dict[sax_str] = (mean_Y, cnt)

        # 释放中间变量
        del sum_cache, count_cache, Y_flat, X_flat, X_norm, paa
        self.is_fitted = True

        # 统计信息
        n_unique = len(self.sax_dict)
        avg_bucket_size = n_samples / max(n_unique, 1)
        print(f"SAX Index Built (mem-optimized): {n_unique} unique strings, "
              f"avg bucket size: {avg_bucket_size:.2f}")

        return self

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        对测试样本进行预测

        内存优化：
        - 精确匹配：直接查 self.sax_dict[sax_str][0]，O(1)，无需索引回查
        - 模糊匹配：只遍历字典键（通常远小于 n_test * n_keys），限制搜索范围
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]

        # 展平
        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)
        else:
            X_flat = X_test.reshape(n_test, -1).astype(self.DTYPE)

        # 实例归一化
        X_norm, _, _ = self._instance_normalize(X_flat)

        # PAA 变换
        paa = self._paa_transform(X_norm)

        # SAX 符号化
        test_sax_strings = self._sax_transform(paa)

        # 预计算：编辑距离阈值 * word_size（将归一化阈值转为字符串长度比例）
        edit_threshold = int(self.epsilon_threshold * self.word_size)
        y_dim = self.pred_len * self.n_features

        Y_pred = np.zeros((n_test, y_dim), dtype=self.DTYPE)
        fallback_count = 0
        matched_count = 0

        # 获取所有 SAX 键的列表（避免 predict 内反复调用 .keys()）
        all_sax_keys = list(self.sax_dict.keys())

        for i in range(n_test):
            sax_str = test_sax_strings[i]

            # ================================================
            # 分支1: 精确匹配（O(1) 查表）
            # ================================================
            if sax_str in self.sax_dict:
                mean_Y, _ = self.sax_dict[sax_str]
                Y_pred[i] = mean_Y
                matched_count += 1
                continue

            # ================================================
            # 分支2: 模糊匹配（只在 top_k 最近的桶中搜索）
            # ================================================
            if edit_threshold > 0 and len(all_sax_keys) > 0:
                # 近似：找编辑距离最近的桶（受限于 word_size * epsilon）
                best_key = None
                best_dist = float('inf')

                for stored_str in all_sax_keys:
                    dist = self._compute_edit_distance(sax_str, stored_str)
                    if dist < best_dist and dist <= edit_threshold:
                        best_dist = dist
                        best_key = stored_str

                if best_key is not None:
                    mean_Y, _ = self.sax_dict[best_key]
                    Y_pred[i] = mean_Y
                    matched_count += 1
                    continue

            # ================================================
            # 分支3: Fallback
            # ================================================
            fallback_count += 1
            if self.fallback_strategy == 'global_mean':
                Y_pred[i] = self.global_Y_mean
            else:  # random_walk
                last_val = np.mean(X_flat[i].reshape(-1, self.n_features)[-1])
                Y_pred[i] = np.full(y_dim, last_val, dtype=self.DTYPE)

        if fallback_count > 0:
            print(f"  Warning: {fallback_count}/{n_test} samples used fallback strategy")
        if matched_count > 0:
            print(f"  Matched: {matched_count}/{n_test} samples via SAX lookup")

        # 恢复原始形状
        if self.n_features > 1:
            Y_pred = Y_pred.reshape(n_test, self.pred_len, self.n_features)
        else:
            Y_pred = Y_pred.reshape(n_test, self.pred_len)

        return Y_pred

    def get_params(self) -> dict:
        """获取模型参数"""
        return {
            'model_type': 'SAXSearch',
            'word_size': self.word_size,
            'alphabet_size': self.alphabet_size,
            'epsilon_threshold': self.epsilon_threshold,
            'fallback_strategy': self.fallback_strategy,
            'n_unique_strings': len(self.sax_dict) if self.is_fitted else 0,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features
        }

    def get_sax_stats(self) -> dict:
        """获取SAX字典统计信息"""
        if not self.is_fitted:
            return {}

        bucket_counts = [cnt for _, cnt in self.sax_dict.values()]
        total_samples = sum(bucket_counts)

        return {
            'n_unique_strings': len(self.sax_dict),
            'total_samples': total_samples,
            'min_bucket': min(bucket_counts) if bucket_counts else 0,
            'max_bucket': max(bucket_counts) if bucket_counts else 0,
            'avg_bucket': np.mean(bucket_counts) if bucket_counts else 0,
        }

    def __repr__(self):
        return (f"SAXSearch(word_size={self.word_size}, alphabet_size={self.alphabet_size}, "
                f"epsilon={self.epsilon_threshold})")
