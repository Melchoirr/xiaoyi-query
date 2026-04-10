import numpy as np


def _safe_denom(x: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    return np.where(np.abs(x) < eps, np.nan, x)


def MAE(pred, true):
    return float(np.mean(np.abs(pred - true)))


def MSE(pred, true):
    return float(np.mean((pred - true) ** 2))


def RMSE(pred, true):
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred, true, eps: float = 1e-2):
    v = np.abs((pred - true) / _safe_denom(true, eps=eps))
    return float(np.nanmean(v))


def MSPE(pred, true, eps: float = 1e-2):
    v = ((pred - true) / _safe_denom(true, eps=eps)) ** 2
    return float(np.nanmean(v))


def SMAPE(pred, true, eps: float = 1e-2):
    denom = np.abs(pred) + np.abs(true)
    denom = np.where(denom < eps, np.nan, denom)
    v = 2.0 * np.abs(pred - true) / denom
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


def per_channel_metrics(pred: np.ndarray, true: np.ndarray, eps: float = 1e-2):
    if pred.ndim != 3:
        raise ValueError("per_channel_metrics expects [B,L,C]")
    c = pred.shape[2]
    out = []
    for i in range(c):
        pi = pred[:, :, i]
        ti = true[:, :, i]
        out.append({
            "channel": i,
            "mae": MAE(pi, ti),
            "mse": MSE(pi, ti),
            "rmse": RMSE(pi, ti),
            "mape": MAPE(pi, ti, eps=eps),
            "smape": SMAPE(pi, ti, eps=eps),
        })
    return out


def metric(pred, true, mape_eps: float = 1e-2):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true, eps=mape_eps)
    mspe = MSPE(pred, true, eps=mape_eps)
    smape = SMAPE(pred, true, eps=mape_eps)
    return mae, mse, rmse, mape, mspe, smape


def metric_dict(pred, true, mape_eps: float = 1e-2):
    mae, mse, rmse, mape, mspe, smape = metric(pred, true, mape_eps=mape_eps)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "smape": smape,
        "rse": RSE(pred, true),
        "corr": CORR(pred, true),
    }
