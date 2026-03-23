# [F01] 预测模型

> 最后更新：2026-03-22 17:30 (UTC+8)

## 概述
时序预测模型层，包含三个不同范式的模型：DLinear（纯线性基线）、PatchTST（Patch-based Transformer）、Sundial（预训练基础模型 zero-shot 推理）。共享 PatchEmbedding 层供 PatchTST 使用。所有模型统一输入 [B, L, C]，输出 [B, pred_len, C]。

## 实现
- **状态**：✅已完成
- **核心文件**：
  - `forecast/models/DLinear.py:32` — Model 类，趋势-季节分解 + 线性映射。moving_avg(kernel=25) 提取趋势，残差为季节，分别线性映射到 pred_len 后相加。支持 `--individual` 切换 CI/CD。
  - `forecast/models/PatchTST.py:6` — Model 类，Channel-Independent Patch Transformer。PatchEmbedding → nn.TransformerEncoder (e_layers 层) → Flatten + Linear head。默认 patch_len=16, stride=8, d_model=128, n_heads=8。
  - `forecast/models/Sundial.py:6` — Model 类，HuggingFace pipeline 封装。懒加载 `thuml/sundial-base-128m`，forward() 抛 NotImplementedError，仅支持 predict()。逐通道逐样本调用 pipeline，返回 numpy。
  - `forecast/layers/Embed.py:6` — PositionalEncoding（正弦/余弦），PatchEmbedding（ReplicationPad1d → unfold → Linear → PE → Dropout）
- **实现方式**：
  - 三模型通过 `exp_basic.py` 的 `model_dict` 统一管理，按 `--model` 参数选择
  - DLinear/PatchTST 支持训练+测试，Sundial 仅 zero-shot 测试
  - Channel-Independent 策略：PatchTST 在 PatchEmbedding 中 reshape [B,L,C]→[B*C,L]；Sundial 在 predict() 中双重循环
- **设计决策**：ADR-001, ADR-002, ADR-003, ADR-005

## 测试
### 测试方法
```bash
# DLinear 快速训练+测试
python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1

# PatchTST 快速训练+测试
python -m forecast.run --model PatchTST --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1

# Sundial zero-shot 测试（需下载模型，首次较慢）
python -m forecast.run --model Sundial --data ETTh1 --pred_len 96 --is_training 0
```

### 测试结果
| 日期 (UTC+8) | 方法 | 结果 | 备注 |
|--------------|------|------|------|
| （暂无记录） | | | |

## 问题跟踪
### 已知问题
| 问题 | 优先级 | 发现日期 | 状态 |
|------|--------|----------|------|
| Sundial 逐样本逐通道推理效率低（双重 for 循环） | P2 | 2026-03-22 | 📋待处理 |
| Sundial pipeline 输出格式可能随 transformers 版本变化 | P2 | 2026-03-22 | 📋待处理 |
| Sundial 首次运行需下载 HF 模型，无离线模式 | P3 | 2026-03-22 | ⚠️设计如此 |

### 解决记录
| 问题 | 解决方案 | 解决日期 | 验证 |
|------|----------|----------|------|

## 时间线
| 时间 (UTC+8) | 事件 | commit |
|--------------|------|--------|
| 2026-03-22 | 初始实现（DLinear + PatchTST + Sundial + Embed） | prediction-fusion 分支 |
