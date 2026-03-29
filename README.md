# 时序预测基线模型

基于记忆检索的时序预测，包含七种算法：
- **传统检索**：PatternSearch (KNN)、LSHSearch (LSH)、SAXSearch (SAX)
- **v3.0 新增**：DTWSearch (DTW 弹性对齐)、MatrixProfileSearch (矩阵轮廓)、TS2VecSearch (深度对比学习)、RAGSearch (Cross-Attention 记忆网络)

## 安装

```bash
# 核心依赖
pip install numpy pandas scikit-learn scipy rich streamlit plotly torch psutil tqdm matplotlib pandas

# 时序专用库（推荐安装，提升对应模型性能）
pip install tslearn stumpy faiss-cpu

# 可选依赖（TS2VecSearch 无 faiss 时的 fallback）
pip install scipy
```

## 重要更新 (v3.0 Dual-Dimension RevIN + 静态可视化)

| 变更类型 | 变更内容 |
|---------|---------|
| **v3.1 新增模型** | 新增 DTWSearch（DTW Sakoe-Chiba）、MatrixProfileSearch（stumpy MASS）、TS2VecSearch（Dilated CNN + faiss）、RAGSearch（Cross-Attention 端到端）共 4 个模型；MODEL_REGISTRY 扩展至 7 个模型；命令行新增 --dtw_radius, --hidden_dim, --rag_d_model 等专属超参数 |
| **Dual-Dimension RevIN（v3.0 新增）** | `--revin_type` 支持 `none` / `temporal` / `feature` / `dual` 四种模式；`temporal` 对齐时间维度，`feature` 对齐特征维度，`dual` 先 feature 再 temporal；所有路径含 std < 1e-5 数值安全防御 |
| **目录规范（v3.0 新增）** | 每个实验存入独立文件夹 `results/{exp_id}/`，内含 `params.json`、`metrics.json`、`preds.npy`、`trues.npy`、`X_test.npy`、`visualization.png` |
| **静态可视化（v3.0 新增）** | 实验结束时自动调用 `plot_comparison_samples` 生成四段线对比 PNG（Historical Lookback / Hist.Pred / Test Input / Pred vs True）；`plot_summary_bar` 生成指标柱状图 |
| **Summary CSV（v3.0 新增）** | `ExperimentRunner` 结束后生成 `results/summary_metrics.csv`，含所有实验的 MAE/MSE/RMSE/MAPE/RSE/CORR 参数与耗时 |
| **实验日志持久化（v3.0 新增）** | `scripts/run_experiments.sh` 使用 `tee` 将每条实验 stdout + stderr 同步写入 `results/logs/{exp_id}.log` |
| **Dashboard 保留** | Streamlit Dashboard 仍可通过 `--dashboard` 启动，用于交互式探查 |
| **RevIN 数值安全（v2.9）** | `std < 1e-5` 时强制置 1.0，避免常量序列除零放大灾难 |
| **全物理尺度落盘（v2.9）** | history / trues / preds 三路 JSON 数据全部 inverse_transform 为原始物理尺度 |

## 使用方法

### 统一入口 `run.py`

```bash
# 基本用法
python run.py --model PatternSearch                          # 单模型
python run.py --model all                                     # 所有模型（推荐，串行）
python run.py --model PatternSearch,LSHSearch                 # 指定模型

# 参数网格
python run.py --model all --seq_len 96 192 --pred_len 24 48 96

# 并行 + 仪表盘（自动内存保护，内存 > 85% 时回退串行）
python run.py --model all --parallel --dashboard

# RevIN（Dual-Dimension 可逆实例归一化）— v3.0 新增
# revin_type: none=无归一化, temporal=时间维度, feature=特征维度, dual=先 feature 再 temporal
python run.py --model PatternSearch --revin_type temporal --seq_len 96 --pred_len 48
python run.py --model all --revin_type dual --seq_len 96 --pred_len 48
python run.py --model all --revin_type feature --dashboard

# 常用 seq_len / pred_len（TSLib 标准配置）
python run.py --model all --seq_len 96 --pred_len 96
python run.py --model all --seq_len 192 --pred_len 192
python run.py --model all --seq_len 336 --pred_len 96
python run.py --model all --seq_len 720 --pred_len 96

# 参数扫描（Shell 脚本 + tee 日志持久化）
bash scripts/run_experiments.sh
bash scripts/run_experiments.sh --models PatternSearch,LSHSearch --revin-types dual,temporal --seq-lens 96,192 --pred-lens 48,96

# 仅启动仪表盘
python run.py --skip_run --dashboard
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | PatternSearch | 模型: PatternSearch / LSHSearch / SAXSearch / DTWSearch / MatrixProfileSearch / TS2VecSearch / RAGSearch / all |
| `--seq_len` | 96 | 输入序列长度（TSLib 标准：96, 192, 336, 720） |
| `--pred_len` | 48 | 预测序列长度（TSLib 标准：96, 192, 336, 720） |
| `--features` | M | M=多变量, S=单变量 |
| `--revin_type` | none | **v3.0 新增**：归一化类型 `none` / `temporal` / `feature` / `dual` |
| `--parallel` | False | 启用并行计算（内存保护自动降级） |
| `--n_workers` | 4 | 并行 worker 数（最大 4） |
| `--use_gpu` | False | `torch.cuda` 可用时，推理使用 GPU |
| `--dashboard` | False | 运行后启动 Streamlit Dashboard |
| `--skip_run` | False | 仅启动仪表盘，跳过实验 |

> **RevIN Type 说明**：
> - `none`：无归一化，原始物理尺度训练
> - `temporal`：Instance Norm，对每个样本沿 axis=1（时间维度）归一化
> - `feature`：Channel Norm，对每个样本沿 axis=-1（特征维度）归一化，对齐多变量量级
> - `dual`：先 feature 再 temporal，预测后先反 temporal 再反 feature，量级最稳定

### 模型特定参数

```bash
# PatternSearch
python run.py --model PatternSearch --top_k 5 --weighted True

# LSHSearch
python run.py --model LSHSearch --n_hash_funcs 16 --n_tables 4 --candidate_cap_total 1024

# SAXSearch
python run.py --model SAXSearch --word_size 8 --alphabet_size 8 --bucket_top_k 8

# DTWSearch (v3.0)
python run.py --model DTWSearch --top_k 5 --dtw_radius 5 --weighted True

# MatrixProfileSearch (v3.0)
python run.py --model MatrixProfileSearch --top_k 5 --mp_normalize True

# TS2VecSearch (v3.0)
python run.py --model TS2VecSearch --hidden_dim 64 --ts2vec_epochs 10 --batch_size 128 --top_k 5

# RAGSearch (v3.0)
python run.py --model RAGSearch --rag_d_model 32 --rag_n_heads 4 --rag_epochs 10 --batch_size 128
```

## 输出结构

```
results/
├── summary_metrics.csv          # v3.0 新增：所有实验汇总（MAE/MSE/RMSE/MAPE/RSE/CORR）
├── experiment_log.json          # 完整 JSON 日志
├── logs/                        # v3.0 新增：每实验独立日志
│   ├── {exp_id}_seq96_pred48_Rd.log
│   └── ...
└── {exp_id}/                    # v3.0 新增：每实验独立文件夹
    ├── params.json              # 完整配置快照
    ├── metrics.json            # MAE/MSE/RMSE/MAPE/RSE/CORR
    ├── preds.npy                # 预测值（原始物理尺度）
    ├── trues.npy                # 真实值（原始物理尺度）
    ├── X_test.npy               # 测试集输入
    └── visualization.png         # v3.0 新增：四段线对比图
```

## 静态可视化（plotting.py）

实验结束时自动调用 `plot_comparison_samples` 生成 `visualization.png`，也可独立使用：

```python
from plotting import plot_comparison_samples, plot_summary_bar

# 生成四段线对比网格图
plot_comparison_samples(
    history=np.load('results/{exp_id}/X_test.npy'),
    preds=np.load('results/{exp_id}/preds.npy'),
    trues=np.load('results/{exp_id}/trues.npy'),
    seq_len=96, pred_len=48, n_features=7,
    model_name='PatternSearch',
    params={'revin_type': 'dual', 'top_k': 5},
    save_path='results/{exp_id}/visualization.png',
    n_samples=9,
    feat_idx=-1   # -1 = Target (OT) 列
)

# 从 summary_metrics.csv 生成指标柱状图
plot_summary_bar('results/summary_metrics.csv', metric='MAE',
                save_path='results/summary_MAE_bar.png')
```

### 四段线说明

| 区域 | 颜色 | 说明 |
|------|------|------|
| A: Hist. Lookback | 蓝灰 | 模型检索到的最相似历史子序列 |
| B: Hist. Prediction | 橙黄 | 历史匹配段对应的真实后续 |
| C: Test Input | 深蓝 | 当前测试样本的输入序列 |
| D: True (Future) | 绿色 | 测试集真实未来值 |
| D: Prediction | 红色虚线 | 模型预测值 |

X 轴统一为时间偏移（0 = 预测起点），Y 轴为原始物理尺度。

## 参数扫描（scripts/run_experiments.sh）

```bash
# 默认：3 模型 × 2 seq × 2 pred × 4 revin × 3 top_k = 144 组合
bash scripts/run_experiments.sh

# 自定义参数
bash scripts/run_experiments.sh \
    --models PatternSearch,LSHSearch \
    --revin-types dual,temporal \
    --seq-lens 96,192,336 \
    --pred-lens 48,96 \
    --top-k 5,10 \
    --parallel

# 仅预览命令不执行
bash scripts/run_experiments.sh --dry-run
```

脚本会自动：
1. 生成 `results/logs/{exp_id}.log`（tee 持久化）
2. 实验结束后汇总 `summary_metrics.csv`
3. 打印 MAE 排名并生成柱状图

## 项目结构

```
.
├── run.py                      # 统一入口（Dual-Dimension RevIN、目录规范、实验日志持久化）
├── plotting.py                 # v3.0 新增：四段线静态可视化 + Summary 柱状图
├── models/
│   ├── PatternSearch.py        # KD-Tree KNN（float32 + torch.cdist + gc）
│   ├── LSHSearch.py            # 局部敏感哈希（uint64 打包 + 两阶段重排）
│   ├── SAXSearch.py            # 符号聚合近似（NearestNeighbors 模糊匹配）
│   ├── DTWSearch.py            # DTW 弹性对齐（Sakoe-Chiba 累积 DP，GPU/CPU 双路径，v3.2）
│   ├── MatrixProfileSearch.py  # 矩阵轮廓（stumpy 工业级，Numba 多核，float32，v3.2）
│   ├── TS2VecSearch.py         # 深度对比学习（Dilated CNN + faiss 向量检索，v3.0）
│   └── RAGSearch.py            # Cross-Attention 记忆网络（端到端训练，v3.0）
├── data_provider/
│   └── data_loader.py          # 数据加载（TSLib 固定边界 + float32）
├── dashboard/
│   └── app.py                  # Streamlit 可视化（全物理尺度、特征选择、HTML flex 指标）
├── scripts/
│   └── run_experiments.sh      # v3.0 新增：参数扫描 + tee 日志 + tqdm 进度
└── results/                    # 实验输出（v3.0 独立文件夹规范）
    ├── summary_metrics.csv     # v3.0：所有实验汇总 CSV
    ├── experiment_log.json    # 完整 JSON 日志
    ├── logs/                  # v3.0：每实验 {exp_id}.log
    └── {exp_id}/
        ├── params.json         # v3.0：配置快照
        ├── metrics.json        # MAE/MSE/RMSE/MAPE/RSE/CORR
        ├── preds.npy           # 预测值（原始物理尺度）
        ├── trues.npy           # 真实值（原始物理尺度）
        ├── X_test.npy          # 测试集输入
        └── visualization.png    # v3.0：四段线对比网格图
```

## Dual-Dimension RevIN（v3.0）

Dual-Dimension RevIN 在 `run_single_experiment` 内部实现，对输入 X 和目标 Y 均适用：

### 四种模式

| 模式 | 公式 | 适用场景 |
|------|------|---------|
| `none` | 无归一化 | 物理尺度直接训练 |
| `temporal` | \(X' = (X - \mu_{axis=1}) / \sigma_{axis=1}\) | 消除序列内均值偏移（TSLib SOTA 基线） |
| `feature` | \(X' = (X - \mu_{axis=-1}) / \sigma_{axis=-1}\) | 对齐多变量不同特征量级 |
| `dual` | 先 feature 再 temporal | 量级差异大且趋势明显的复杂数据 |

### 数值安全

所有四种模式的 std 计算均含防御机制：

```python
def _safe_std(std, threshold=1e-5):
    return np.where(std < threshold, 1.0, std)  # 防止除零放大
```

### 指标计算

指标（MAE/MSE/RMSE/MAPE/RSE/CORR）始终在**归一化空间**计算，与 TSLib 0.3 学术基线量级对齐。

### 为什么用 RevIN

| 维度 | 无归一化 | Mean-Shift | RevIN temporal | RevIN dual |
|------|---------|------------|-----------------|-------------|
| 均值对齐 | ❌ | ✅ | ✅ | ✅ |
| 方差对齐 | ❌ | ❌ | ✅ | ✅ |
| 特征量级对齐 | ❌ | ❌ | ❌ | ✅ |
| TSLib SOTA 量级 | ❌ | 部分 | ✅ | ✅ |

## 算法对比

| 模型 | 搜索精度 | 检索速度 | 特点 |
|------|---------|---------|------|
| PatternSearch | 精确 | O(log n) | 欧氏距离，逆距离加权，GPU 加速 |
| LSHSearch | 近似 | O(1) | 随机投影，uint64 打包，两阶段重排 |
| SAXSearch | 模糊 | O(n) | PAA 降维，NearestNeighbors 模糊匹配 |
| DTWSearch | 弹性对齐 | O(chunk·n·m·r) | **v3.2: GPU 严格 Sakoe-Chiba 累积 DP**（无 4D 张量，显存 ≈410MB），**CPU: tslearn 精确 DTW**，双路径 chunked 防 OOM |
| MatrixProfileSearch | 精确子序列 | O(n·log n) | **v3.2: stumpy 工业级库**（Numba 多核加速），MASS z-normalized；**DTYPE=np.float32**；全程 chunked 防止 32GB 溢出 |
| TS2VecSearch | 深度表示 | O(n) | Dilated CNN 对比编码，faiss 向量库极速检索（v3.0） |
| RAGSearch | 端到端 | O(n) | Cross-Attention 记忆网络，MSE 端到端训练（v3.0） |

## 内存优化（v2.0）

本项目针对 8核32G 环境下的 OOM 问题进行了系统性优化，理论上可将峰值内存从 30GB+ 压制到 **4~8GB**。

### 优化措施概览

| 层级 | 文件 | 优化手段 | 预期内存收益 |
|------|------|---------|------------|
| 数据加载 | `data_provider/data_loader.py` | float64→float32，预分配，del+gc | ~50% 降幅 |
| 实验调度 | `run.py` | 实验间强制 gc.collect()，JSON 截断100条 | ~1~2GB |
| 并行保护 | `run.py` | 内存>85%回退串行，max_workers=4 | 避免峰值叠加 |
| SAX 索引 | `models/SAXSearch.py` | 哈希桶存均值+NearestNeighbors | 10~100x 压缩 |
| LSH 索引 | `models/LSHSearch.py` | 哈希桶存均值+候选重排 | 10~100x 压缩 |
| NPY 落盘 | `run.py` | 完整预测只存.npy，不进JSON | JSON 体积从 MB→KB |

### 1. 数据加载（float32 + 预分配）

```python
# data_loader.py
DTYPE = np.float32

# CSV 读取后立即转换为 float32
raw = df_data[cols_data].values.astype(DTYPE)

# 预分配而非 append 列表再转 np.array
X_all = np.empty((n_samples, seq_len, n_feature), dtype=DTYPE)
```

### 2. 哈希桶预聚合（SAX / LSH）

每个桶只存均值而非原始索引列表，压缩比可达 10~100x。

### 3. 实验调度 GC + JSON 截断

每次实验后强制 `gc.collect()`；JSON 中只保留前 100 条预览，完整数据落盘 `.npy`。

### 4. 并行内存保护

内存 > 85% 时自动回退串行；并行 worker 数限制为 4。

## 内存估算参考

以 ETTm1 数据集，`seq_len=96, pred_len=48, features=M`（7 特征）为例：

| 数据结构 | float64 原始 | float32 优化后 |
|---------|-------------|---------------|
| 训练集 X [~26K, 96, 7] | ~55 MB | ~28 MB |
| 训练集 Y [~26K, 48, 7] | ~27 MB | ~14 MB |
| 单模型峰值合计 | **~350 MB** | **~100 MB** |
| 3模型串行峰值 | ~1 GB | ~300 MB |
| 3模型并行峰值（优化前） | **~30 GB (OOM!)** | **~4 GB (安全)** |

## 仪表盘（Dashboard）

启动方式：

```bash
streamlit run dashboard/app.py
# 或
python run.py --skip_run --dashboard
```

仪表盘功能：

| 功能 | 说明 |
|------|------|
| **全物理尺度绘图（v2.9 新增）** | JSON preview 三路数据全部 inverse_transform 为原始物理尺度，Dashboard 直接绘制，无前端计算 |
| **彻底解决缓存脏读（v2.9 新增）** | `time.sleep(1)` + `st.cache_data.clear()` + `st.rerun()` 三连保障 |
| **特征维度选择器** | 动态下拉框，支持 ETT 预定义名称（HUFL/HULL/MUFL/.../OT）；末列标注 (Target) |
| **HTML flex 单行指标** | `display: flex` 替代 `st.columns()`，跨屏幕绝对单行 |
| **Plotly zeroline** | 所有图表 Y 轴 `zeroline=True, zerolinecolor='lightgray'` |

> **推荐**：v3.0 优先使用静态 `visualization.png` 和 `summary_metrics.csv` 进行分析，Dashboard 用于交互式深度探查。

## 输出（v3.0 目录规范）

每个实验自动生成独立文件夹 `results/{exp_id}/`：

| 文件 | 说明 |
|------|------|
| `params.json` | 完整配置快照（v3.0 新增） |
| `metrics.json` | MAE/MSE/RMSE/MAPE/RSE/CORR（v3.0 新增） |
| `preds.npy` | 预测值（原始物理尺度，float32） |
| `trues.npy` | 真实值（原始物理尺度，float32） |
| `X_test.npy` | 测试集输入（用于 Dashboard 连贯波形） |
| `visualization.png` | 四段线对比网格图（v3.0 新增） |

汇总文件：
- `results/summary_metrics.csv` — 所有实验的指标汇总
- `results/experiment_log.json` — 完整 JSON 日志
- `results/logs/{exp_id}.log` — 每实验独立日志

运行 `--dashboard` 后访问 `http://localhost:8501` 查看可视化（Streamlit Dashboard）。

## 核心模块复用

```python
from run import run_single_experiment, ExperimentRunner

# 单独运行一个实验（v3.0 revin_type 参数）
result = run_single_experiment({
    'model_name': 'PatternSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5,
    'revin_type': 'temporal',   # none / temporal / feature / dual
})

# Dual RevIN 模式
result = run_single_experiment({
    'model_name': 'SAXSearch',
    'seq_len': 96,
    'pred_len': 96,
    'word_size': 8,
    'alphabet_size': 8,
    'revin_type': 'dual',
})

# ── v3.0 新增模型使用示例 ───────────────────────────────────

# DTWSearch（DTW 弹性对齐）
result = run_single_experiment({
    'model_name': 'DTWSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5,
    'dtw_radius': 5,
    'weighted': True,
    'revin_type': 'temporal',
})

# MatrixProfileSearch（矩阵轮廓）
result = run_single_experiment({
    'model_name': 'MatrixProfileSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5,
    'normalize': True,
    'revin_type': 'temporal',
})

# TS2VecSearch（深度对比学习 + faiss）
result = run_single_experiment({
    'model_name': 'TS2VecSearch',
    'seq_len': 96,
    'pred_len': 48,
    'hidden_dim': 64,
    'epochs': 10,
    'batch_size': 128,
    'top_k': 5,
    'revin_type': 'temporal',
})

# RAGSearch（Cross-Attention 记忆网络）
result = run_single_experiment({
    'model_name': 'RAGSearch',
    'seq_len': 96,
    'pred_len': 48,
    'd_model': 32,
    'n_heads': 4,
    'epochs': 10,
    'batch_size': 128,
    'revin_type': 'temporal',
})

# 返回值含 exp_id / exp_dir / metrics / preview

# 批量运行（自动生成 summary_metrics.csv）
runner = ExperimentRunner(args)
runner.run()

# 静态绘图（独立使用）
from plotting import plot_comparison_samples, plot_summary_bar
plot_comparison_samples(
    history=np.load('results/{exp_id}/X_test.npy'),
    preds=np.load('results/{exp_id}/preds.npy'),
    trues=np.load('results/{exp_id}/trues.npy'),
    seq_len=96, pred_len=48, n_features=7,
    model_name='PatternSearch',
    params={'revin_type': 'dual', 'top_k': 5},
    save_path='results/{exp_id}/visualization.png',
    n_samples=9, feat_idx=-1
)
plot_summary_bar('results/summary_metrics.csv', metric='MAE')
```

## 版本历史

| 版本 | 更新内容 |
|------|---------|
| **v3.2** | DTWSearch: GPU 严格 Sakoe-Chiba 累积 DP（无 4D 张量，≈410MB 显存），tslearn CPU 精确路径，predict_chunk 分块；MatrixProfileSearch: stumpy 工业级集成（MASS z-normalized），DTYPE=np.float32，全程 chunked 防止 32GB 溢出；所有模型 GPU 自动检测；进度日志输出 |
| **v3.1** | 新增 4 个前沿模型：DTWSearch、MatrixProfileSearch、TS2VecSearch（RAGSearch Siamese 架构）；MODEL_REGISTRY 扩展至 7 个模型 |
| **v3.0** | Dual-Dimension RevIN（temporal/feature/dual）、目录规范化（results/{exp_id}/）、静态可视化（plotting.py 四段线网格图）、summary_metrics.csv、scripts/run_experiments.sh 参数扫描 + tee 日志 |
| **v2.9** | RevIN 数值安全（std < 1e-5 强制置 1.0）、全物理尺度 JSON 落盘 |
| **v2.1** | Streamlit Dashboard 交互式探查、auto-refresh 增强 |
| **v2.0** | 内存优化（chunked 处理、gc.collect()、float32）、并行实验 |
| **v1.0** | PatternSearch / LSHSearch / SAXSearch 基线实现 |
