"""
时序数据预处理器 - 实现实例级Z-Score归一化
"""

from typing import List, Tuple
import numpy as np


class TSProcessor:
    """时序数据预处理器，负责归一化和反归一化"""

    def __init__(self):
        """初始化时序处理器"""
        self._mu: float = 0.0
        self._sigma: float = 1.0

    def normalize(self, sequence: List[float]) -> Tuple[np.ndarray, float, float]:
        """
        对输入序列进行Z-Score归一化

        Args:
            sequence: 输入的原始时序数据列表

        Returns:
            Tuple[np.ndarray, float, float]:
                - normalized: 归一化后的numpy数组
                - mu: 均值
                - sigma: 标准差
        """
        sequence_array = np.array(sequence, dtype=np.float64)

        mu = np.mean(sequence_array)
        sigma = np.std(sequence_array, ddof=0)

        if sigma < 1e-10:
            sigma = 1.0

        normalized = (sequence_array - mu) / sigma

        return normalized, mu, sigma

    def denormalize(
        self, sequence: np.ndarray, mu: float, sigma: float
    ) -> np.ndarray:
        """
        将归一化后的序列还原为真实量级

        Args:
            sequence: 归一化后的numpy数组
            mu: 原始均值
            sigma: 原始标准差

        Returns:
            np.ndarray: 反归一化后的真实量级数组
        """
        if isinstance(sequence, list):
            sequence = np.array(sequence, dtype=np.float64)

        denormalized = sequence * sigma + mu

        return denormalized
