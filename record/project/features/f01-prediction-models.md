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
  - `forecast/models/CosineMatch.py` — MSE最小匹配基线，从训练集检索最近邻作为预测
  - `forecast/models/PlotFusion.py` — 多模型预测对比可视化
- **功能描述**：多范式时序预测模型。可训练模型：DLinear（趋势-季节分解+线性映射）、PatchTST（Patch+TransformerEncoder）。基础模型 zero-shot：Sundial（HF pipeline）、Chronos（Amazon chronos-bolt-small）、Timer（THU timer-base-84m, AutoModelForCausalLM）、TimesFM（Google timesfm-2.0-500m）。非参数基线：CosineMatch（MSE最小匹配，train集使用leave-one-out避免信息泄露）。可视化：PlotFusion（多模型对比图，参数化支持任意模型组合）。所有模型统一输入 [B,L,C] 输出 [B,pred_len,C]，通过 exp_basic.py 的 model_dict 统一管理。
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

### [修改] 2026-03-31 16:00 — CosineMatch/PlotFusion 从 scripts 迁移至 models，修复信息泄露

<details><summary>详情</summary>

**计划**：将 `cosine_match.py` 和 `plot_fusion.py` 从 `forecast/scripts/` 迁移到 `forecast/models/`，修复 CosineMatch 的多个 bug，并将两者集成到 `run.py` 统一入口。

**代码修改**：
- `forecast/scripts/cosine_match.py` → `forecast/models/CosineMatch.py`：
  - **修复 train 集信息泄露**：新增 `exclude_self` 参数，当 flag='train' 时排除自身匹配（leave-one-out），避免 stacking 过拟合
  - **修复 RESULT 打印 bug**：单独记录 test 集 metrics，不再依赖循环末尾变量
  - 改用 `forecast.utils.metrics.metric()` 统一指标计算
- `forecast/scripts/plot_fusion.py` → `forecast/models/PlotFusion.py`：
  - 参数化重构：所有硬编码配置（seq_len、models、paths）改为 argparse 参数
  - 支持任意模型组合和动态颜色分配
- 删除 `forecast/scripts/cosine_match.py` 和 `forecast/scripts/plot_fusion.py`

**测试**：
| 方法 | 结果 | 备注 |
|------|------|------|
| `from forecast.models.CosineMatch import ...` | ✅ | 导入正常 |
| `from forecast.models.PlotFusion import main` | ✅ | 导入正常 |

</details>

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
