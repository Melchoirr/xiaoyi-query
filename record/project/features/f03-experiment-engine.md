# [F03] 实验引擎

## 状态

- **实现状态**：✅已完成
- **核心文件**：
  - `forecast/exp/exp_basic.py:6` — Exp_Basic 基类（model_dict、设备管理）
  - `forecast/exp/exp_long_term_forecasting.py:14` — Exp_Long_Term_Forecast（train/vali/test）
  - `forecast/utils/tools.py:5` — EarlyStopping + adjust_learning_rate
- **功能描述**：训练/验证/测试完整实验循环。train(): Adam+MSELoss，epoch 循环，EarlyStopping(patience=3)，LR 调度(每 epoch 减半)，保存 best checkpoint。test(): 加载 checkpoint（Sundial 直接 predict），计算 MAE/MSE 等指标，保存 npy。Sundial 跳过训练。
- **测试方法**：
  ```bash
  python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 3
  ls ./checkpoints/DLinear_ETTh1_M_sl96_pl96/checkpoint.pth
  ls ./forecast/results/DLinear_ETTh1_M_sl96_pl96/
  ```

## 变化

### [实现] 2026-03-22 17:30 — 初始实现 (`90d939e`)

<details><summary>详情</summary>

**计划**：实现标准训练/验证/测试循环，支持三个模型统一调度，Sundial 特殊路径。
**代码修改**：
- 新增 `exp/exp_basic.py`：Exp_Basic 基类，model_dict 管理三模型，CUDA/MPS/CPU 设备
- 新增 `exp/exp_long_term_forecasting.py`：train() 完整训练循环 + vali() + test()（含 Sundial 分支）
- 新增 `utils/tools.py`：EarlyStopping + adjust_learning_rate (type1/type2)

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| （暂无） | ⚪未测试 | |

**已知问题**：
- 实验结果未汇总对比（每次单独 npy，无统一对比表）— P2

</details>
