# [F04] 评估指标

> 最后更新：2026-03-22 17:30 (UTC+8)

## 概述
时序预测评估指标集合。提供 MAE、MSE、RMSE、MAPE、MSPE 五个标准指标和 RSE、CORR 两个辅助指标。metric() 函数是被实验引擎调用的统一入口。

## 实现
- **状态**：✅已完成
- **核心文件**：
  - `forecast/utils/metrics.py:1` — 全部指标函数
- **实现方式**：
  - 输入：pred 和 true 均为 numpy array，形状 [N, pred_len, C]
  - MAE: `mean(|pred - true|)`
  - MSE: `mean((pred - true)²)`
  - RMSE: `√MSE`
  - MAPE: `mean(|pred - true| / true)` — ⚠️ true=0 时除零
  - MSPE: `mean(((pred - true) / true)²)` — ⚠️ true=0 时除零
  - metric() 返回 (mae, mse, rmse, mape, mspe) 元组

## 测试
### 测试方法
```python
import numpy as np
from forecast.utils.metrics import metric
pred = np.random.randn(100, 96, 7)
true = np.random.randn(100, 96, 7) + 2  # 偏移避免除零
mae, mse, rmse, mape, mspe = metric(pred, true)
print(f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
```

### 测试结果
| 日期 (UTC+8) | 方法 | 结果 | 备注 |
|--------------|------|------|------|
| （暂无记录） | | | |

## 问题跟踪
### 已知问题
| 问题 | 优先级 | 发现日期 | 状态 |
|------|--------|----------|------|
| MAPE/MSPE 在 true 值含 0 时产生 inf/nan（除零） | P2 | 2026-03-22 | 📋待处理 |

### 解决记录
| 问题 | 解决方案 | 解决日期 | 验证 |
|------|----------|----------|------|

## 时间线
| 时间 (UTC+8) | 事件 | commit |
|--------------|------|--------|
| 2026-03-22 | 初始实现 | prediction-fusion 分支 |
