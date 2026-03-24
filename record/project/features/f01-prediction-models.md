# [F01] 预测模型

## 状态

- **实现状态**：✅已完成
- **核心文件**：
  - `forecast/models/DLinear.py:32` — Model 类，趋势-季节分解 + 线性映射
  - `forecast/models/PatchTST.py:6` — Model 类，Channel-Independent Patch Transformer
  - `forecast/models/Sundial.py:6` — Model 类，HuggingFace pipeline zero-shot 推理
  - `forecast/models/Chronos.py` — Model 类，Amazon Chronos zero-shot 推理
  - `forecast/models/Timer.py` — Model 类，THU Timer zero-shot 推理
  - `forecast/models/TimesFM.py` — Model 类，Google TimesFM zero-shot 推理
  - `forecast/layers/Embed.py:22` — PatchEmbedding（Patch 切分 + 投影 + 位置编码）
- **功能描述**：六个不同范式的时序预测模型。可训练模型：DLinear（趋势-季节分解+线性映射）、PatchTST（Patch+TransformerEncoder）。基础模型 zero-shot：Sundial（HF pipeline）、Chronos（Amazon chronos-bolt-small）、Timer（THU timer-base-84m, AutoModelForCausalLM）、TimesFM（Google timesfm-2.0-500m）。所有模型统一输入 [B,L,C] 输出 [B,pred_len,C]，通过 exp_basic.py 的 model_dict 统一管理。
- **测试方法**：
  ```bash
  python -m forecast.run --model DLinear --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1
  python -m forecast.run --model PatchTST --data ETTh1 --pred_len 96 --is_training 1 --train_epochs 1
  python -m forecast.run --model Sundial --data ETTh1 --pred_len 96 --is_training 0
  python -m forecast.run --model Chronos --data ETTh1 --pred_len 96 --is_training 0
  python -m forecast.run --model Timer --data ETTh1 --pred_len 96 --is_training 0
  python -m forecast.run --model TimesFM --data ETTh1 --pred_len 96 --is_training 0
  ```

## 变化

### [实现] 2026-03-24 — 新增 Chronos/Timer/TimesFM 三个基础模型

<details><summary>详情</summary>

**计划**：扩展基础模型覆盖，新增 Amazon Chronos、THU Timer、Google TimesFM 三个 zero-shot 模型。
**代码修改**：
- 新增 `forecast/models/Chronos.py`：BaseChronosPipeline.from_pretrained()，predict_quantiles() 提取中位数
- 新增 `forecast/models/Timer.py`：AutoModelForCausalLM，generate() 逐通道推理
- 新增 `forecast/models/TimesFM.py`：timesfm.TimesFm()，支持频率指示映射
- 修改 `forecast/exp/exp_basic.py`：model_dict 新增 Chronos/Timer/TimesFM

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| （暂无） | ⚪未测试 | |

</details>

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
