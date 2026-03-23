# 设计决策记录

## 格式说明

采用 ADR (Architecture Decision Record) 简化版。

状态标签：`提议` / `已采纳` / `已废弃`

---

## ADR-001 — 三模型对比架构（DLinear + PatchTST + Sundial）

- **日期**：2026-03-22
- **状态**：已采纳
- **背景**：需要在 ETT 数据集上对比不同范式的时序预测模型。
- **决策**：选择 DLinear（纯线性）、PatchTST（Patch Transformer）、Sundial（预训练基础模型）三个代表性模型。
- **理由**：DLinear 代表"线性模型足够好"（AAAI 2023）；PatchTST 是 Patch 机制 Transformer 代表（ICLR 2023）；Sundial 代表大模型 zero-shot 范式（THU-ML）。三者覆盖当前主要技术路线。
- **代价**：Sundial 依赖 HuggingFace transformers 和模型下载。
- **影响**：models/ 下三个独立模型文件，exp_basic.py 通过 model_dict 统一管理。

## ADR-002 — Channel-Independent 策略

- **日期**：2026-03-22
- **状态**：已采纳
- **背景**：多变量时序中变量间关系的建模方式是核心设计选择。
- **决策**：PatchTST 和 Sundial 采用 CI；DLinear 支持 CI/CD 切换（`--individual`）。
- **理由**：CI 在中等规模数据集上通常优于 CD（PatchTST 论文结论）；减少参数量和过拟合风险。
- **代价**：无法捕获变量间相关性。
- **影响**：PatchEmbedding reshape [B,L,C]→[B*C,L]；Sundial 逐通道调用 pipeline。

## ADR-003 — Sundial Zero-Shot 不训练

- **日期**：2026-03-22
- **状态**：已采纳
- **背景**：Sundial 是预训练基础模型，支持 zero-shot 预测。
- **决策**：run.py 中 Sundial 跳过训练，直接 test；forward() 抛 NotImplementedError。
- **理由**：保持 zero-shot 特性；与训练模型的对比更有意义。
- **代价**：无法通过微调适配特定数据集；test() 需特殊分支。
- **影响**：exp_long_term_forecasting.py test() 中 Sundial 调用 predict() 而非 forward()。

## ADR-004 — ETT 标准数据分割（12/4/8 月）

- **日期**：2026-03-22
- **状态**：已采纳
- **背景**：ETT 数据集的 train/val/test 分割需与学术基准一致。
- **决策**：train 前12月, val 中间4月, test 后8月。
- **理由**：与所有 ETT 基准论文一致，结果可直接对比。
- **代价**：硬编码分割边界，无法灵活调整。
- **影响**：data_loader.py 中 border1s/border2s 固定值。

## ADR-005 — DLinear 趋势-季节分解

- **日期**：2026-03-22
- **状态**：已采纳
- **背景**：DLinear 用简单线性层替代 Transformer，需要归纳偏置捕获时序特性。
- **决策**：Moving Average（kernel=25）分解趋势和季节，分别线性映射后相加。
- **理由**：分解后分量更易被线性层建模；kernel_size=25 与原论文一致。
- **代价**：kernel_size 对不同数据集可能不最优。
- **影响**：DLinear.py 中 series_decomp + moving_avg 类。
