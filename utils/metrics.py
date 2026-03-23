"""
评估指标模块
实现时序预测常用的评估指标：MAE, MSE, RMSE, MAPE, MSPE
"""

import numpy as np
from typing import Union


def mae(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE)

    Args:
        preds: 预测值数组
        trues: 真实值数组

    Returns:
        MAE值
    """
    return np.mean(np.abs(preds - trues))


def mse(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Mean Squared Error (MSE)

    Args:
        preds: 预测值数组
        trues: 真实值数组

    Returns:
        MSE值
    """
    return np.mean((preds - trues) ** 2)


def rmse(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE)

    Args:
        preds: 预测值数组
        trues: 真实值数组

    Returns:
        RMSE值
    """
    return np.sqrt(mse(preds, trues))


def mape(preds: np.ndarray, trues: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Mean Absolute Percentage Error (MAPE)

    避免除零问题，使用 epsilon 进行平滑

    Args:
        preds: 预测值数组
        trues: 真实值数组
        epsilon: 平滑常数，避免除零

    Returns:
        MAPE值 (百分比形式，如 10.5 表示 10.5%)
    """
    # 避免除零和极端值
    trues_safe = np.where(np.abs(trues) < epsilon, epsilon, trues)
    return np.mean(np.abs((trues - preds) / trues_safe)) * 100


def mspe(preds: np.ndarray, trues: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Mean Squared Percentage Error (MSPE)

    避免除零问题，使用 epsilon 进行平滑

    Args:
        preds: 预测值数组
        trues: 真实值数组
        epsilon: 平滑常数，避免除零

    Returns:
        MSPE值 (百分比形式)
    """
    # 避免除零和极端值
    trues_safe = np.where(np.abs(trues) < epsilon, epsilon, trues)
    return np.mean(((trues - preds) / trues_safe) ** 2) * 100


def smape(preds: np.ndarray, trues: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)

    更对称的MAPE变体

    Args:
        preds: 预测值数组
        trues: 真实值数组
        epsilon: 平滑常数

    Returns:
        SMAPE值 (百分比形式)
    """
    numerator = np.abs(preds - trues)
    denominator = (np.abs(preds) + np.abs(trues)) / 2
    denominator_safe = np.where(denominator < epsilon, epsilon, denominator)
    return np.mean(numerator / denominator_safe) * 100


def calculate_all_metrics(preds: np.ndarray, trues: np.ndarray,
                         prefix: str = '') -> dict:
    """
    计算所有评估指标

    Args:
        preds: 预测值数组，任意形状
        trues: 真实值数组，任意形状
        prefix: 指标名称前缀

    Returns:
        包含所有指标的字典
    """
    # 展平为1D数组进行计算
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()

    metrics = {
        f'{prefix}MAE': mae(preds_flat, trues_flat),
        f'{prefix}MSE': mse(preds_flat, trues_flat),
        f'{prefix}RMSE': rmse(preds_flat, trues_flat),
        f'{prefix}MAPE': mape(preds_flat, trues_flat),
        f'{prefix}MSPE': mspe(preds_flat, trues_flat),
    }

    return metrics


def print_metrics(metrics: dict, decimals: int = 6):
    """
    格式化打印指标

    Args:
        metrics: 指标字典
        decimals: 小数位数
    """
    print("\n" + "=" * 50)
    print("Evaluation Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if 'MAPE' in key or 'MSPE' in key:
            print(f"{key}: {value:.{decimals}f}%")
        else:
            print(f"{key}: {value:.{decimals}f}")
    print("=" * 50)


def save_metrics_to_file(metrics: dict, filepath: str, mode: str = 'a'):
    """
    将指标保存到文件

    Args:
        metrics: 指标字典
        filepath: 文件路径
        mode: 写入模式，'a'追加，'w'覆盖
    """
    with open(filepath, mode) as f:
        if mode == 'w':
            f.write("=" * 60 + "\n")
            f.write("PatternSearch Baseline Evaluation Results\n")
            f.write("=" * 60 + "\n")

        for key, value in metrics.items():
            if 'MAPE' in key or 'MSPE' in key:
                f.write(f"{key}: {value:.6f}%\n")
            else:
                f.write(f"{key}: {value:.6f}\n")

        f.write("-" * 60 + "\n")
