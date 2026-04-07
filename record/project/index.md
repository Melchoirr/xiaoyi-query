# xiaoyi-query (prediction-fusion) 功能索引

> 最后更新：2026-03-31 (UTC+8)

## 架构概览

时序预测基准测试与融合框架，在 ETT 数据集上对比三个范式的模型。

```
CLI 入口 (forecast/run.py) + 批量脚本 (scripts/train/)
    ↓
实验引擎 (forecast/exp/)
    ├── 训练/验证/测试循环
    ├── EarlyStopping + LR 调度
    ↓              ↓
数据管道           预测模型
├── ETT Dataset    ├── forecast/models/ — DLinear / PatchTST / PrimitiveFusion (可训练)
├── DataLoader     ├── forecast/baselines/ — CosineMatch / GuidedMatch / PredMatch (非参数)
└── 时间特征编码   ├── scripts/benchmark/ — Sundial/Chronos/Timer/TimesFM (zero-shot)
    ↓              └── forecast/fusion/ — XGBoost Stacking 融合
评估指标 (MAE/MSE/RMSE/MAPE/MSPE)
    ↓
产物 → outputs/results/ (数值) + outputs/figures/ (图片)
```

## 功能清单

| ID | 功能 | 核心文件 | 最初实现 | 最后变更 | 状态 | 维护 | 详情 |
|----|------|----------|----------|----------|------|------|------|
| F01 | 预测模型 | models/*.py, layers/Embed.py | 03-22 | 03-31 16:00 | ⚠️有注意事项 | 🟢在用 | [详情](features/f01-prediction-models.md) |
| F02 | 数据管道 | data_provider/*.py, utils/timefeatures.py | 03-22 | 03-22 | ✅无误 | 🟢在用 | [详情](features/f02-data-pipeline.md) |
| F03 | 实验引擎 | exp/*.py, utils/tools.py | 03-22 | 03-24 | ⚠️有注意事项 | 🟢在用 | [详情](features/f03-experiment-engine.md) |
| F04 | 评估指标 | utils/metrics.py | 03-22 | 03-22 | ⚠️有注意事项 | 🟢在用 | [详情](features/f04-metrics.md) |
| F05 | CLI 与批量实验 | run.py, scripts/run_all.sh | 03-22 | 03-31 16:00 | ✅无误 | 🟢在用 | [详情](features/f05-cli-and-scripts.md) |
| F06 | XGBoost Stacking 融合 | fusion/stacking.py | 03-24 | 03-24 | 🔧进行中 | 🟢在用 | [详情](features/f06-xgb-fusion.md) |

### 状态图例
- ✅无误：功能正常，无已知问题
- ⚠️有注意事项：可用但有需要注意的地方
- 🔴有Bug：存在已确认的 Bug
- ❌不可用：功能不可用

### 维护图例
- 🟢在用：活跃维护
- 🟡冻结：不再迭代但仍在使用
- 🔴弃用：已被替代或移除

## 全局问题汇总

> 以下汇总自各功能详情文件的已知问题，更新时从 features/*.md 中聚合，不单独维护。

| 问题 | 优先级 | 来源 | 状态 |
|------|--------|------|------|
| 测试覆盖为零（无 tests/ 目录） | P1 | 全局 | 📋待处理 |
| 无 pyproject.toml 或 requirements.txt 依赖声明 | P2 | 全局 | 📋待处理 |
| MAPE/MSPE 在 true=0 时除零 | P2 | [F04](features/f04-metrics.md) | 📋待处理 |
| Sundial 逐样本推理效率低 | P2 | [F01](features/f01-prediction-models.md) | 📋待处理 |
| 实验结果无汇总对比表 | P2 | [F03](features/f03-experiment-engine.md) | 📋待处理 |
| Sundial 首次需下载 HF 模型 | P3 | [F01](features/f01-prediction-models.md) | ⚠️设计如此 |
