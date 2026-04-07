# 实验大纲

> 最后更新：2026-04-06

状态图例：✅已完成 | 🔧进行中 | ❌缺失待补充 | 📋计划中

---

## 1. ETT 数据集基本分析

### 1.1 可视化

| 内容 | 代码 | 产物 | 状态 |
|------|------|------|------|
| 原始 7 变量时序图 + train/val/test 分割线 | `scripts/plotting/plot_etth1_raw.py` | `outputs/figures/data_analysis/etth1_raw_series.png` | ✅ |
| StandardScaler 归一化后序列 | `scripts/plotting/plot_etth1_normalized.py` | `outputs/figures/data_analysis/etth1_normalized_series.png` | ✅ |
| 不同时间窗口切片特征 | `scripts/plotting/plot_etth1_slices.py` | `outputs/figures/data_analysis/etth1_slices/`(6 页) | ✅ |

### 1.2 频域 / 统计分析

| 内容 | 代码 | 产物 | 状态 |
|------|------|------|------|
| FFT / PSD 功率谱密度 | `scripts/plotting/plot_etth1_spectral.py` | `outputs/figures/data_analysis/etth1_psd.png` | ✅ |
| 变量间互相关分析 | 同上 | `outputs/figures/data_analysis/etth1_xcorr.png` | ✅ |
| 变量间交叉谱分析 | 同上 | `outputs/figures/data_analysis/etth1_cross_spectral.png` | ✅ |
| SNR 信噪比分析 | — | — | ❌缺失 |
| Lag 滞后分析 | — | — | ❌缺失 |

---

## 2. DLinear 7x7 跨变量预测

### 2.1 直接交叉预测 (49 组 src→tgt)

| 内容 | 代码 | 产物 | 状态 |
|------|------|------|------|
| 7 变量 x 7 变量 DLinear 单变量预测 | `scripts/train/run_cross_var.sh` | `outputs/results/DLinear_ETTh1_crossvar_src{0-6}_tgt{0-6}_sl192_pl96/` | ✅ |

- 模型：DLinear (enc_in=1, CI 模式)
- 运行入口：`forecast/run.py --mode cross_var`
- 配置：epochs=15, patience=5, lr=0.005, batch=32

### 2.2 XGBoost 跨变量融合

| 内容 | 代码 | 产物 | 状态 |
|------|------|------|------|
| CrossVarStacking (7 source → 1 target) | `forecast/fusion/cross_var_stacking.py` | `outputs/results/CrossVarFusion_*_ETTh1_crossvar_sl192_pl96/` | ✅ |

- 运行入口：`forecast/run.py --mode cross_var_fusion`
- 对每个目标变量，将 7 个 source channel 的 DLinear 预测作特征训练 XGBoost

### 2.3 预测残差 + XGBoost

| 内容 | 代码 | 产物 | 状态 |
|------|------|------|------|
| ResidualStacking (self_pred + 残差学习) | `forecast/fusion/cross_var_stacking.py` | `outputs/results/ResidualFusion_*_ETTh1_crossvar_sl192_pl96/` | ✅ |

- 运行入口：`forecast/run.py --mode residual_fusion`

### 2.4 可视化

| 内容 | 产物 | 状态 |
|------|------|------|
| 7x7 MSE 热力图 | `outputs/figures/cross_var/cross_var_7x7_heatmap.png` | ✅ |
| 跨变量对比图 | `outputs/figures/cross_var/cross_var_comparison.png` | ✅ |
| 跨变量细节图 | `outputs/figures/cross_var/cross_var_detail.png` | ✅ |

---

## 3. 基础模型对比

### 3.1 可训练模型

| 模型 | 代码 | 配置 | 状态 |
|------|------|------|------|
| DLinear | `forecast/models/DLinear.py` | 趋势-季节分解 + 线性映射；seq_len=96/336/512, pred_len=96；lr=0.005, epochs=10 | ✅ |
| PatchTST | `forecast/models/PatchTST.py` | CI Patch Transformer + RevIN；lr=0.0001, epochs=100, patience=100 | ✅ |

### 3.2 零样本基础模型

统一入口：`scripts/benchmark/benchmark.py`

| 模型 | seq_len 配置 | 状态 |
|------|-------------|------|
| Chronos | 96 / 336 / 512 | ✅ |
| Timer | 96 / 336 / 512 | ✅ |
| Sundial | 96 / 512 | ✅ |
| TimesFM | 96 / 336 | ✅ |
| Moirai | 512 | ❌日志存在但结果缺失 |
| TimeMoE | — | 📋计划中 |
| TiRex | — | 📋计划中 |

---

## 4. 多变量原语化 (PrimitiveFusion)

| 内容 | 代码 | 产物 | 状态 |
|------|------|------|------|
| 模型：K 原语码本 + Cross-Attention | `forecast/models/PrimitiveFusion.py` | — | ✅ |
| 训练脚本 | `scripts/train/run_train.sh` | `outputs/results/PrimitiveFusion_ETTh1_M_sl{96,512}_pl96/` | ✅ |
| 分析：逐通道指标 | `scripts/analysis/analyze_primitive_fusion.py` | `outputs/figures/primitive/per_channel_mse.png` | ✅ |
| 分析：Codebook PCA | 同上 | `outputs/figures/primitive/codebook_pca.png` | ✅ |
| 分析：Codebook 相似度热力图 | 同上 | `outputs/figures/primitive/codebook_similarity.png` | ✅ |
| 分析：原语分配热力图 | 同上 | `outputs/figures/primitive/assignment_heatmap.png` | ✅ |
| 分析：原语利用率 (entropy) | 同上 | `outputs/figures/primitive/primitive_utilization.png` | ✅ |

**结论**：信号机制弱

关键参数：`--num_primitives 16`, `--primitive_temp 1.0`, `--n_cross_layers 1`

---

## 5. 匹配方法

### 5.1 CosineMatch (历史序列 MSE 最小匹配)

| 内容 | 代码 | 状态 |
|------|------|------|
| 逐维度 Top-K MSE 加权匹配 | `forecast/baselines/CosineMatch.py` | ✅ |

- train 集使用 leave-one-out 避免信息泄露
- 多 seq_len 实验：48 / 96 / 192 / 512
- 运行入口：`forecast/run.py --mode cosine_match`

### 5.2 GuidedMatch (预测引导匹配)

| 内容 | 代码 | 状态 |
|------|------|------|
| DLinear/PatchTST 预测作 query，在 train 集真实未来中检索 | `forecast/baselines/GuidedMatch.py` | ✅ |

- 核心假设：更接近目标的 query 能检索到更好的邻居
- 运行入口：`forecast/run.py --mode cosine_match --model GuidedMatch`

### 5.3 PredMatch (Oracle 上界)

| 内容 | 代码 | 状态 |
|------|------|------|
| 用真实 pred_len 在训练集 pred_len 中匹配（理论上界） | `forecast/baselines/PredMatch.py` | ✅ |

- 衡量"训练集中是否存在相似的未来模式"

### 5.4 匹配分析

| 内容 | 代码 | 状态 |
|------|------|------|
| SeqMatch vs PredMatch 统计特征对比（均值/标准差/趋势/自相关） | `scripts/analysis/analyze_match_stats.py` | ✅ |
| 时间周期相位对齐分析 (24h 日周期 / 168h 周周期) | `scripts/analysis/analyze_phase_alignment.py` | ✅ |
| Top-K 匹配可视化 (7x3 子图) | `scripts/plotting/plot_cosine_topk.py` | ✅ |
| SeqMatch vs PredMatch 对比图 | `scripts/plotting/plot_seq_vs_pred_match.py` | ✅ |
| 逐通道指标对比 (SeqMatch/GuidedMatch/DLinear/PredMatch) | `scripts/analysis/per_channel_metrics.py` | ✅ |

---

## 6. 多模型结果调制 (XGBoost Stacking)

| 内容 | 代码 | 产物 | 状态 |
|------|------|------|------|
| XGBStacking 融合 | `forecast/fusion/stacking.py` | `outputs/results/XGBFusion_ETTh1_M_sl{96,336}_pl96/` | ✅ |
| 融合对比可视化 | `scripts/plotting/plot_fusion.py` | `outputs/results/fusion_comparison.png` | ✅ |
| 日志解析汇总 | `scripts/analysis/parse_logs.py` | CSV 汇总表 | ✅ |

- 特征：各模型的原始预测值 + horizon_idx
- 逐通道训练独立 XGBRegressor(n_estimators=100, max_depth=4)
- 参与模型：DLinear, PatchTST, Chronos, Timer, Sundial
- 运行入口：`forecast/run.py --mode fusion`

---

## 7. 调研

不在仓库中记录。涉及方向：TESS, TimeXer, RLinear, 预测的生成模式, 归一化。

TESS 论文：`docs/papers/TESS_2026.pdf`

---

## 8. 理论

不在仓库中记录。涉及方向：Lipschitz 常数等。

---

## 附：实验矩阵

```
维度 1: 实验模式 (run.py --mode)
  ├── single           → 单模型训练/测试
  ├── fusion           → XGBoost Stacking 多模型融合
  ├── cosine_match     → 匹配方法 (CosineMatch/GuidedMatch/PredMatch)
  ├── plot             → 融合对比可视化
  ├── cross_var        → 跨变量预测
  ├── cross_var_fusion → 跨变量 XGBoost 融合
  └── residual_fusion  → 残差学习融合

维度 2: 模型类型
  ├── 可训练: DLinear, PatchTST, PrimitiveFusion
  ├── 零样本: Chronos, Timer, Sundial, TimesFM, Moirai
  └── 非参数: CosineMatch, GuidedMatch, PredMatch

维度 3: 数据集
  ├── ETTh1 (小时级, 主要实验)
  └── ETTh2, ETTm1, ETTm2 (支持但多数未测)

维度 4: 融合策略
  ├── XGBoost Stacking (多模型 → 单预测)
  ├── CrossVar Fusion (多变量 → 单变量)
  └── Residual Fusion (自身预测 + 残差学习)
```

## 附：批量运行脚本

| 脚本 | 内容 | 位置 |
|------|------|------|
| run_all.sh | DLinear + PatchTST 训练 → CosineMatch → GuidedMatch → XGBoost 融合 → 对比图 | `scripts/train/run_all.sh` |
| run_cross_var.sh | 7x7 跨变量 DLinear 实验 + CrossVarFusion + ResidualFusion | `scripts/train/run_cross_var.sh` |
| run_train.sh | PrimitiveFusion 单独训练 | `scripts/train/run_train.sh` |
