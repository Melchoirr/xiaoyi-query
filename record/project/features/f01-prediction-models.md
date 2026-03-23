# [F01] 预测模型

## 状态

- **实现状态**：✅已完成
- **核心文件**：
  - `forecast/models/DLinear.py:32` — Model 类，趋势-季节分解 + 线性映射
  - `forecast/models/PatchTST.py:6` — Model 类，Channel-Independent Patch Transformer
  - `forecast/models/Sundial.py:6` — Model 类，HuggingFace pipeline zero-shot 推理
  - `forecast/layers/Embed.py:22` — PatchEmbedding（Patch 切分 + 投影 + 位置编码）
- **功能描述**：三个不同范式的时序预测模型。DLinear 用 moving_avg(kernel=25) 做趋势-季节分解后线性映射；PatchTST 用 Patch + TransformerEncoder；Sundial 通过 HF pipeline 做 zero-shot channel-independent 推理。所有模型统一输入 [B,L,C] 输出 [B,pred_len,C]，通过 exp_basic.py 的 model_dict 统一管理。
- **测试方法**：
  ```bash
  python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1
  python -m forecast.run --model PatchTST --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1
  python -m forecast.run --model Sundial --data ETTh1 --pred_len 96 --is_training 0
  ```

## 变化

### [实现] 2026-03-22 17:30 — 初始实现 (`90d939e`)

<details><summary>详情</summary>

**计划**：实现 DLinear + PatchTST + Sundial 三模型对比框架，覆盖线性、Transformer、基础模型三个范式。
**代码修改**：
- 新增 `models/DLinear.py`：moving_avg + series_decomp + Model，支持 --individual CI/CD 切换
- 新增 `models/PatchTST.py`：PatchEmbedding → TransformerEncoder → Linear head
- 新增 `models/Sundial.py`：HF pipeline 懒加载，predict() 逐通道逐样本推理
- 新增 `layers/Embed.py`：PositionalEncoding + PatchEmbedding

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| （暂无） | ⚪未测试 | |

**已知问题**：
- Sundial 逐样本逐通道推理效率低（双重 for 循环）— P2
- Sundial pipeline 输出格式可能随 transformers 版本变化 — P2
- Sundial 首次运行需下载 HF 模型（~500MB）— P3，设计如此

</details>
