# [F04] 评估指标

## 状态

- **实现状态**：✅已完成
- **核心文件**：
  - `forecast/utils/metrics.py:1` — MAE/MSE/RMSE/MAPE/MSPE/RSE/CORR + metric() 入口
- **功能描述**：时序预测标准评估指标。metric(pred, true) 返回 (mae, mse, rmse, mape, mspe)。输入 numpy array [N, pred_len, C]。
- **测试方法**：
  ```python
  import numpy as np
  from forecast.utils.metrics import metric
  pred = np.random.randn(100, 96, 7)
  true = np.random.randn(100, 96, 7) + 2
  mae, mse, rmse, mape, mspe = metric(pred, true)
  print(f"MAE={mae:.4f}, MSE={mse:.4f}")
  ```

## 变化

### [实现] 2026-03-22 17:30 — 初始实现 (`90d939e`)

<details><summary>详情</summary>

**计划**：实现标准时序预测评估指标集合。
**代码修改**：新增 `utils/metrics.py`，7 个指标函数 + metric() 统一入口。

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| （暂无） | ⚪未测试 | |

**已知问题**：
- MAPE/MSPE 在 true=0 时除零产生 inf/nan — P2

</details>
