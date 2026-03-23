# 小易时序预测融合框架 (xiaoyi-query / prediction-fusion)

## 项目概要
时序预测基准测试与融合框架，在 ETT 数据集上对比 DLinear、PatchTST、Sundial 三个模型。
架构：CLI 入口 (run.py) → 实验引擎 (exp/) → 模型层 (models/) + 数据层 (data_provider/)
Python 3.13 / PyTorch / HuggingFace Transformers (Sundial)

## 文档体系
开始工作前阅读相关文档了解当前状态：
- `record/project/index.md` — 功能索引总览（状态、时间、维护情况）
- `record/project/features/` — 各功能详情文件（实现/测试/问题/时间线）
- `record/project/decisions.md` — 设计决策 (ADR)

### Skill 使用
- `/catch-up` — 初始化文档或追赶落后的文档状态（扫描代码+git→生成/更新所有文档）
- `/update-docs` — 日常文档更新（Plan阶段标记计划，Execute阶段根据diff记录实现）
- `/audit` — 审计未测试区域和潜在问题，输出优先级排序的待办清单

## 开发命令
- 训练+测试单模型: `python -m forecast.run --model DLinear --data ETTh1 --pred_len 96`
- 测试 Sundial (zero-shot): `python -m forecast.run --model Sundial --data ETTh1 --pred_len 96 --is_training 0`
- 全量基准测试: `bash forecast/scripts/run_all.sh`
- 安装依赖: `uv sync`

## 关键文件
| 文件 | 职责 |
|------|------|
| forecast/run.py | CLI 入口 + argparse 配置 |
| forecast/exp/exp_basic.py | 实验基类（设备管理、模型构建） |
| forecast/exp/exp_long_term_forecasting.py | 训练/验证/测试循环 |
| forecast/models/DLinear.py | DLinear 模型（趋势-季节分解） |
| forecast/models/PatchTST.py | PatchTST 模型（Patch + Transformer） |
| forecast/models/Sundial.py | Sundial 基础模型（HF pipeline 推理） |
| forecast/data_provider/data_loader.py | ETT 数据集加载器（小时/分钟） |
| forecast/data_provider/data_factory.py | 数据集工厂 + DataLoader 构建 |
| forecast/layers/Embed.py | PatchEmbedding + PositionalEncoding |
| forecast/utils/metrics.py | MAE/MSE/RMSE/MAPE/MSPE 指标 |
| forecast/utils/tools.py | EarlyStopping + 学习率调度 |
| forecast/utils/timefeatures.py | 时间特征编码 |
| forecast/scripts/run_all.sh | 全量实验批处理脚本 |

## 行为规则
1. 使用 `python` 而非 `python3`
2. 运行模型用 `python -m forecast.run`（模块模式），不要 `python forecast/run.py`
3. Sundial 是 zero-shot 模型，不需要训练（`--is_training 0`）
4. checkpoints/ 在 .gitignore 中，不提交
5. ETT_data/ 下的 CSV 文件是数据集，不修改
