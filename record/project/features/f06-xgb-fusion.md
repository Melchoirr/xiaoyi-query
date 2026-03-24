# [F06] XGBoost Stacking 融合

## 状态

- **实现状态**：🔧进行中
- **核心文件**：
  - `forecast/fusion/__init__.py` — 模块初始化
  - `forecast/fusion/stacking.py` — XGBStacking 类（加载预测、构造特征、训练、推理评估）
- **功能描述**：用 XGBoost 做 Stacking 融合。加载各基模型在 val 集上的预测作为训练特征（+ horizon_idx），训练 XGBRegressor 学习最优组合权重，然后在 test 集上推理并评估。特征维度 = 模型数 + 1（horizon 位置）。结果保存到 `forecast/results/XGBFusion_{data}_{features}_sl{seq_len}_pl{pred_len}/`。
- **测试方法**：
  ```bash
  # 先跑基模型并保存 val 预测
  python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --save_val_pred
  python -m forecast.run --model PatchTST --data ETTh1 --pred_len 96 --is_training 1 --save_val_pred
  # 运行融合
  python -m forecast.run --mode fusion --data ETTh1 --pred_len 96 --fusion_models DLinear,PatchTST
  ```

## 变化

### [实现] 2026-03-24 — 初始实现

<details><summary>详情</summary>

**计划**：用 XGBoost Stacking 融合多个基模型的预测结果，提升整体预测精度。
**代码修改**：
- 新增 `forecast/fusion/__init__.py`：空模块
- 新增 `forecast/fusion/stacking.py`：XGBStacking 类
  - `_load_predictions(flag)`：加载各模型预测，stack 为 [N, pred_len, C, num_models]
  - `_build_features(stacked)`：展平 + 添加 horizon_idx 特征
  - `train()`：val 集训练 XGBRegressor(n_estimators=100, max_depth=4)
  - `predict_and_evaluate()`：test 集推理 + metric 评估 + 保存 npy
- 修改 `forecast/run.py`：新增 `--mode fusion` 入口
- 修改 `forecast/scripts/run_all.sh`：新增融合步骤

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| （暂无） | ⚪未测试 | 需先跑基模型生成 val/test 预测 |

</details>
