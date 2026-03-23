"""
SAXSearch: 基于符号聚合近似(Symbolic Aggregate approXimation)的时序预测基线模型
将时序压缩为"字符串"进行模糊匹配，实现高效的序列检索与预测

核心思想：
- PAA (Piecewise Aggregate Approximation): 将长序列降维为短片段
- 高斯分位数映射: 将每个PAA段映射到字母符号
- 字符串匹配: 相似波形产生相似字符串
- 编辑距离: 允许模糊匹配
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
    4. 存储: 将符号字符串作为键，对应Y值作为值
    5. 查询: 将测试序列转为SAX字符串，查找匹配项
    """

    def __init__(
        self,
        word_size: int = 8,
        alphabet_size: int = 8,
        epsilon_threshold: float = 1.0,
        fallback_strategy: str = 'global_mean',
        random_state: Optional[int] = 42
    ):
        """
        Args:
            word_size: PAA片段数量（即SAX字符串长度）
            alphabet_size: 符号表大小（如8则使用'a'-'h'）
            epsilon_threshold: 编辑距离容忍阈值（归一化尺度）
            fallback_strategy: 无匹配时的备选策略
                             - 'global_mean': 使用全局Y均值
                             - 'random_walk': 使用输入序列最后值
            random_state: 随机种子（用于初始化）
        """
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.epsilon_threshold = epsilon_threshold
        self.fallback_strategy = fallback_strategy
        self.random_state = random_state

        # 字母表
        self.alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]

        # 高斯分位数 breakpoints
        self.breakpoints = self._compute_breakpoints()

        # 模型状态
        self.sax_dict: Dict[str, List[int]] = {}  # 字符串 -> 样本索引列表
        self.memory_Y: Optional[np.ndarray] = None
        self.global_Y_mean: Optional[np.ndarray] = None
        self.is_fitted: bool = False

        # 维度信息
        self.seq_len: int = 0
        self.n_features: int = 1
        self.pred_len: int = 0

    def _compute_breakpoints(self) -> np.ndarray:
        """
        计算高斯分布的分位数breakpoints

        将标准正态分布划分为alphabet_size个等概率区间
        返回每个区间的上边界

        Returns:
            breakpoints: shape [alphabet_size - 1]
        """
        # 生成分位数点
        breakpoints = norm.ppf(np.linspace(1.0 / self.alphabet_size, 
                                           (self.alphabet_size - 1) / self.alphabet_size,
                                           self.alphabet_size - 1))
        return breakpoints

    def _instance_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        实例级Z-Score归一化

        Args:
            X: 输入序列 [n_samples, seq_len] 或 [n_samples, seq_len, n_features]

        Returns:
            X_norm: 归一化后的序列
            X_mean: 每个样本的均值
            X_std: 每个样本的标准差
        """
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
        """
        PAA (Piecewise Aggregate Approximation) 变换

        将长度为seq_len的序列降维到word_size个片段

        Args:
            X: 输入序列 [n_samples, seq_len]（已归一化）

        Returns:
            paa: PAA降维后的序列 [n_samples, word_size]
        """
        n_samples, seq_len = X.shape
        paa = np.zeros((n_samples, self.word_size))

        # 每个PAA片段的长度
        segment_size = seq_len / self.word_size

        for i in range(self.word_size):
            # 每个片段的起始和结束位置
            start = int(i * segment_size)
            end = int((i + 1) * segment_size)

            # 对片段取平均（边缘情况：最后一个片段可能更长）
            if i == self.word_size - 1:
                end = seq_len

            # 取该片段的均值
            paa[:, i] = np.mean(X[:, start:end], axis=1)

        return paa

    def _sax_transform(self, paa: np.ndarray) -> List[str]:
        """
        将PAA序列转换为SAX符号串

        Args:
            paa: PAA序列 [n_samples, word_size]

        Returns:
            sax_strings: SAX符号串列表，每个元素如 "abccba"
        """
        n_samples = paa.shape[0]
        sax_strings = []

        for i in range(n_samples):
            symbols = []
            for j in range(self.word_size):
                value = paa[i, j]

                # 根据breakpoints找到对应的符号
                symbol_idx = 0
                for bp in self.breakpoints:
                    if value > bp:
                        symbol_idx += 1

                symbols.append(self.alphabet[symbol_idx])

            sax_strings.append(''.join(symbols))

        return sax_strings

    def _compute_edit_distance(self, s1: str, s2: str) -> int:
        """
        计算两个字符串之间的编辑距离（Levenshtein Distance）

        使用动态规划实现

        Args:
            s1, s2: 两个字符串

        Returns:
            编辑距离
        """
        if len(s1) < len(s2):
            return self._compute_edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        构建SAX索引（记忆库）

        Args:
            X_train: 输入序列数组
                   shape: [n_samples, seq_len, n_features] 或 [n_samples, seq_len]
            Y_train: 目标序列数组
                   shape: [n_samples, pred_len, n_features] 或 [n_samples, pred_len]
        """
        n_samples = X_train.shape[0]

        # 保存维度信息
        if X_train.ndim == 3:
            self.seq_len = X_train.shape[1]
            self.n_features = X_train.shape[2]
            X_flat = X_train.reshape(n_samples, -1)
        else:
            self.seq_len = X_train.shape[1]
            self.n_features = 1
            X_flat = X_train.reshape(n_samples, -1)

        if Y_train.ndim == 3:
            self.pred_len = Y_train.shape[1]
            self.memory_Y = Y_train.reshape(n_samples, -1)
        else:
            self.pred_len = Y_train.shape[1]
            self.memory_Y = Y_train.reshape(n_samples, -1)

        # 计算全局Y均值
        self.global_Y_mean = np.mean(self.memory_Y, axis=0)

        # 实例归一化
        X_norm, _, _ = self._instance_normalize(X_flat)

        # PAA变换
        paa = self._paa_transform(X_norm)

        # SAX符号化
        sax_strings = self._sax_transform(paa)

        # 构建字典
        self.sax_dict = {}
        for i, sax_str in enumerate(sax_strings):
            if sax_str not in self.sax_dict:
                self.sax_dict[sax_str] = []
            self.sax_dict[sax_str].append(i)

        self.is_fitted = True

        # 统计信息
        n_unique = len(self.sax_dict)
        avg_bucket_size = n_samples / max(n_unique, 1)
        print(f"SAX Index Built: {n_unique} unique strings, "
              f"avg bucket size: {avg_bucket_size:.2f}")

        return self

    def predict(self, X_test: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        对测试样本进行预测

        Args:
            X_test: 测试输入序列
                   shape: [n_test, seq_len, n_features] 或 [n_test, seq_len]
            top_k: 最多聚合的样本数量

        Returns:
            Y_pred: 预测输出
                   shape: [n_test, pred_len, n_features] 或 [n_test, pred_len]
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        n_test = X_test.shape[0]

        # 展平
        if X_test.ndim == 3:
            X_flat = X_test.reshape(n_test, -1)
        else:
            X_flat = X_test.reshape(n_test, -1)

        # 实例归一化
        X_norm, _, _ = self._instance_normalize(X_flat)

        # PAA变换
        paa = self._paa_transform(X_norm)

        # SAX符号化
        test_sax_strings = self._sax_transform(paa)

        # 查找匹配
        Y_pred = np.zeros((n_test, self.pred_len * self.n_features), dtype=np.float64)

        fallback_count = 0
        matched_count = 0

        for i in range(n_test):
            sax_str = test_sax_strings[i]

            if sax_str in self.sax_dict:
                # 精确匹配
                candidates = self.sax_dict[sax_str][:top_k]
                candidate_Y = self.memory_Y[candidates]
                Y_pred[i] = np.mean(candidate_Y, axis=0)
                matched_count += 1
            else:
                # 模糊匹配：找编辑距离最近的字符串
                best_matches = []
                best_distance = float('inf')

                for stored_str, indices in self.sax_dict.items():
                    dist = self._compute_edit_distance(sax_str, stored_str)

                    if dist < best_distance:
                        best_distance = dist
                        best_matches = indices
                    elif dist == best_distance:
                        best_matches.extend(indices)

                # 检查距离是否在容忍范围内
                if best_distance <= self.epsilon_threshold and best_matches:
                    best_matches = best_matches[:top_k]
                    candidate_Y = self.memory_Y[best_matches]
                    Y_pred[i] = np.mean(candidate_Y, axis=0)
                    matched_count += 1
                else:
                    # Fallback策略
                    fallback_count += 1
                    if self.fallback_strategy == 'global_mean':
                        Y_pred[i] = self.global_Y_mean
                    else:  # random_walk
                        last_values = X_flat[i].reshape(-1, self.n_features)
                        Y_pred[i] = np.tile(np.mean(last_values[-1]), self.pred_len)

        if fallback_count > 0:
            print(f"  Warning: {fallback_count}/{n_test} samples used fallback strategy")

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
            'n_memory': self.memory_Y.shape[0] if self.is_fitted else 0,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features
        }

    def get_sax_stats(self) -> dict:
        """
        获取SAX字典统计信息

        Returns:
            统计信息字典
        """
        if not self.is_fitted:
            return {}

        bucket_sizes = [len(indices) for indices in self.sax_dict.values()]

        return {
            'n_unique_strings': len(self.sax_dict),
            'min_bucket': min(bucket_sizes) if bucket_sizes else 0,
            'max_bucket': max(bucket_sizes) if bucket_sizes else 0,
            'avg_bucket': np.mean(bucket_sizes) if bucket_sizes else 0,
        }

    def __repr__(self):
        return (f"SAXSearch(word_size={self.word_size}, alphabet_size={self.alphabet_size}, "
                f"epsilon={self.epsilon_threshold})")
