import numpy as np


def _safe_denom(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.where(np.abs(x) < eps, np.nan, x)


def MAE(pred, true):
    return float(np.mean(np.abs(pred - true)))


def MSE(pred, true):
    return float(np.mean((pred - true) ** 2))


def RMSE(pred, true):
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred, true):
    v = np.abs((pred - true) / _safe_denom(true))
    return float(np.nanmean(v))


def MSPE(pred, true):
    v = ((pred - true) / _safe_denom(true)) ** 2
    return float(np.nanmean(v))


def RSE(pred, true):
    denom = np.sqrt(np.sum((true - true.mean()) ** 2))
    if denom < 1e-10:
        return float("nan")
    return float(np.sqrt(np.sum((true - pred) ** 2)) / denom)


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    d = np.where(d < 1e-10, np.nan, d)
    ratio = u / d
    if np.all(np.isnan(ratio)):
        return float("nan")
    return float(np.nanmean(ratio))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe


def metric_dict(pred, true):
    mae, mse, rmse, mape, mspe = metric(pred, true)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "rse": RSE(pred, true),
        "corr": CORR(pred, true),
    }
