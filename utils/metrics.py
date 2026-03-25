"""
评估指标模块 - 对齐 TSLib 学术规范

关键规范：
1. 所有返回值强制 float() 包裹，避免 numpy 标量无法被 json 序列化
2. MAPE / MSPE 移除 *100（TSLib 原始公式不带百分号）
3. 引入 Mask 机制处理接近 0 的真实值，避免数值爆炸
4. 新增 RSE（Root Squared Error）和 CORR（Pearson 相关系数）
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
#  基础指标
# ─────────────────────────────────────────────────────────────

def mae(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Mean Absolute Error
    """
    return float(np.mean(np.abs(preds - trues)))


def mse(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Mean Squared Error
    """
    return float(np.mean((preds - trues) ** 2))


def rmse(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Root Mean Squared Error
    """
    return float(np.sqrt(mse(preds, trues)))


# ─────────────────────────────────────────────────────────────
#  相对误差指标（TSLib 原始公式，无 *100）
#  Mask 机制：过滤掉 |true| < 1e-3 的无效点
# ─────────────────────────────────────────────────────────────

def mape(preds: np.ndarray, trues: np.ndarray, mask_threshold: float = 1e-3) -> float:
    """
    Mean Absolute Percentage Error（TSLib 规范，无 *100）

    Mask 机制：
    - 过滤掉 |true| < mask_threshold 的极小值点（避免数值爆炸）
    - 对其余有效点取平均
    """
    mask = np.abs(trues) > mask_threshold
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs((trues[mask] - preds[mask]) / trues[mask])))


def mspe(preds: np.ndarray, trues: np.ndarray, mask_threshold: float = 1e-3) -> float:
    """
    Mean Squared Percentage Error（TSLib 规范，无 *100）

    Mask 机制：
    - 过滤掉 |true| < mask_threshold 的极小值点
    """
    mask = np.abs(trues) > mask_threshold
    if not np.any(mask):
        return float('nan')
    return float(np.mean(((trues[mask] - preds[mask]) / trues[mask]) ** 2))


def smape(preds: np.ndarray, trues: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Symmetric Mean Absolute Percentage Error
    """
    numerator = np.abs(preds - trues)
    denominator = (np.abs(preds) + np.abs(trues)) / 2
    denominator_safe = np.where(denominator < epsilon, epsilon, denominator)
    return float(np.mean(numerator / denominator_safe) * 100)


# ─────────────────────────────────────────────────────────────
#  TSLib 特有指标：RSE & CORR
# ─────────────────────────────────────────────────────────────

def rse(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Root Squared Error / 归一化 RMSE
    RSE = sqrt(sum((pred - true)^2)) / sqrt(sum((true - mean(true))^2))

    衡量预测值相对于真实值方差的标准化误差。
    值越接近 0 越好（0 表示完美预测），接近 1 与均值预测相当。
    """
    numerator = np.sum((trues - preds) ** 2)
    denominator = np.sum((trues - np.mean(trues)) ** 2)
    if denominator < 1e-10:
        return float('nan')
    return float(np.sqrt(numerator / denominator))


def corr(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Pearson 相关系数
    Corr = Cov(pred, true) / (std(pred) * std(true))

    值域 [-1, 1]，越接近 1 表示正相关越强，越接近 -1 表示负相关。
    """
    preds_flat = preds.ravel()
    trues_flat = trues.ravel()

    preds_mean = np.mean(preds_flat)
    trues_mean = np.mean(trues_flat)

    preds_centered = preds_flat - preds_mean
    trues_centered = trues_flat - trues_mean

    cov = np.sum(preds_centered * trues_centered)
    std_pred = np.sqrt(np.sum(preds_centered ** 2))
    std_true = np.sqrt(np.sum(trues_centered ** 2))

    if std_pred < 1e-10 or std_true < 1e-10:
        return float('nan')

    return float(cov / (std_pred * std_true))


# ─────────────────────────────────────────────────────────────
#  批量计算
# ─────────────────────────────────────────────────────────────

def calculate_all_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    """
    计算全部评估指标（TSLib 规范）
    所有返回值均为原生 Python float（可直接 json 序列化）
    """
    # 展平以便计算
    p = preds.ravel()
    t = trues.ravel()

    return {
        'MAE':  mae(p, t),
        'MSE':  mse(p, t),
        'RMSE': rmse(p, t),
        'MAPE': mape(p, t),
        'MSPE': mspe(p, t),
        'SMAPE': smape(p, t),
        'RSE':  rse(p, t),
        'CORR': corr(p, t),
    }


def print_metrics(metrics: dict, prefix: str = ''):
    """打印指标（保留两位小数）"""
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"{prefix}{key}: N/A")
        else:
            print(f"{prefix}{key}: {value:.6f}")
