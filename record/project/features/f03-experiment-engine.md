# [F03] 实验引擎

> 最后更新：2026-03-22 17:30 (UTC+8)

## 概述
训练/验证/测试的完整实验循环。Exp_Basic 基类管理设备和模型构建，Exp_Long_Term_Forecast 实现 train/vali/test 三个核心方法。配合 EarlyStopping 和学习率调度工具。Sundial 跳过训练直接测试。

## 实现
- **状态**：✅已完成
- **核心文件**：
  - `forecast/exp/exp_basic.py:6` — Exp_Basic 基类，model_dict 管理三个模型，_acquire_device() 支持 CUDA/MPS/CPU
  - `forecast/exp/exp_long_term_forecasting.py:14` — Exp_Long_Term_Forecast
    - `train()`:50 — Adam + MSELoss，每 epoch 遍历 train_loader，每 100 iter 打印 loss，epoch 末计算 vali/test loss，EarlyStopping 检查，LR 调度，最终加载 best model
    - `vali()`:28 — eval + no_grad，计算 MSE 损失
    - `test()`:112 — 加载 checkpoint（或 Sundial 直接推理），收集 preds/trues，计算 MAE/MSE/RMSE/MAPE/MSPE，保存 npy 到 result_path
  - `forecast/utils/tools.py:5` — EarlyStopping（patience 次无改善停止 + 保存 checkpoint），adjust_learning_rate（type1: 每 epoch 减半; type2: 手动阶梯）
- **实现方式**：
  - **features='MS'** 时 f_dim=-1，仅取最后一个变量计算损失
  - **Sundial 分支**：test() 中 `model.predict()` 返回 numpy，其他模型 `model()` 返回 tensor
  - **结果保存**：`{result_path}/{setting}/` 下 pred.npy, true.npy, metrics.npy
  - **checkpoint**：`{checkpoints}/{setting}/checkpoint.pth`
- **设计决策**：ADR-003（Sundial 不训练）

## 测试
### 测试方法
```bash
# 完整训练+测试流程
python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 3

# 验证产出文件
ls ./checkpoints/DLinear_ETTh1_M_sl96_pl96/checkpoint.pth
ls ./forecast/results/DLinear_ETTh1_M_sl96_pl96/{pred,true,metrics}.npy
```

### 测试结果
| 日期 (UTC+8) | 方法 | 结果 | 备注 |
|--------------|------|------|------|
| （暂无记录） | | | |

## 问题跟踪
### 已知问题
| 问题 | 优先级 | 发现日期 | 状态 |
|------|--------|----------|------|
| 结果未汇总对比（每次实验单独 npy，无统一对比表） | P2 | 2026-03-22 | 📋待处理 |

### 解决记录
| 问题 | 解决方案 | 解决日期 | 验证 |
|------|----------|----------|------|

## 时间线
| 时间 (UTC+8) | 事件 | commit |
|--------------|------|--------|
| 2026-03-22 | 初始实现（实验基类 + 训练/测试循环 + 工具） | prediction-fusion 分支 |
