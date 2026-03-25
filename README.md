# 时序预测基线模型

基于记忆检索的时序预测，包含三种算法：**PatternSearch** (KNN)、**LSHSearch** (局部敏感哈希)、**SAXSearch** (符号聚合近似)。

## 安装

```bash
pip install numpy pandas scikit-learn scipy rich streamlit plotly torch psutil tqdm
```

## 重要更新 (v2.8 波形量纲对齐 + 双尺度切换)

v2.8 / v2.7 对齐 TSLib 指标尺度 + RevIN + Dashboard UI 升级：

| 变更类型 | 变更内容 |
|---------|---------|
| **波形量纲对齐（v2.8 新增）** | preview `trues` 改用归一化值，与 `history` / `preds` 三者量纲统一，彻底解决历史波形与未来波形不在同一尺度的问题 |
| **双尺度切换（v2.8 新增）** | `st.checkbox("显示归一化尺度")` 控制是否前端动态归一化；勾选后用 `(X - X_mean) / X_std` 统一 history/trues/preds，以 history 为基准计算统计量 |
| **彻底解决缓存脏读（v2.8 新增）** | 移除 `@st.cache_data`，每次直接读文件系统；`subprocess.Popen` 完成后显式 `st.cache_data.clear()` + `st.rerun()` |
| **特征维度选择器（v2.7 新增）** | 动态下拉框切换任意特征列；支持 ETT 预定义名称（HUFL/HULL/MUFL/.../OT）；末列标注 (Target) |
| **HTML flex 单行指标（v2.7 新增）** | `display: flex` 替代 `st.columns()`，跨屏幕绝对单行 |
| **Plotly zeroline（v2.7 新增）** | 所有图表 Y 轴 `zeroline=True, zerolinecolor='lightgray'`，波形穿越 0 轴清晰可见 |
| **RevIN 数值安全（v2.7 新增）** | `std < 1e-5` 时强制置 1.0，避免常量序列/方差极小数据除零放大 |
| **RevIN（可逆实例归一化）** | `--revin` 开关：训练 `(X-mean)/std`，推理 `Y_pred*std+mean`，SOTA 指标量级 |
| **Mean-Shift → RevIN** | 原 `--mean_shift` 已升级为完整 `--revin`，去均值版本已废弃 |
| **TSLib Y 截断** | 强制 `Y = Y[:, -pred_len:, :]` 消除 `label_len + pred_len` 残留干扰 |
| **表单 expander** | 侧边栏表单折叠 `expanded=False`，节省主视图空间 |
| **实时 Log + 自动刷新** | `subprocess.Popen` + `iter(stdout.readline)` 实时打屏；`st.rerun()` 自动刷新加载最新结果 |
| **评估指标** | MAPE/MSPE 移除 `*100`；新增 RSE、CORR；全部 `float()` 包裹防 JSON 序列化 |

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

# GPU 加速（需安装 CUDA 版 PyTorch；PatternSearch 用 torch.cdist+topk，LSH 批量投影）
python run.py --model PatternSearch --use_gpu --dashboard

# 仅启动仪表盘
python run.py --skip_run --dashboard

# RevIN（可逆实例归一化）— 深度学习 SOTA 预处理方式
python run.py --model PatternSearch --revin --seq_len 96 --pred_len 96
python run.py --model all --revin --dashboard

# 常用 seq_len / pred_len（TSLib 标准配置）
python run.py --model all --seq_len 96 --pred_len 96
python run.py --model all --seq_len 192 --pred_len 192
python run.py --model all --seq_len 336 --pred_len 96
python run.py --model all --seq_len 720 --pred_len 96
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | PatternSearch | 模型: PatternSearch / LSHSearch / SAXSearch / all |
| `--seq_len` | 96 | 输入序列长度（TSLib 标准：96, 192, 336, 720） |
| `--pred_len` | 48 | 预测序列长度（TSLib 标准：96, 192, 336, 720） |
| `--features` | M | M=多变量, S=单变量 |
| `--revin` | False | **启用 RevIN（可逆实例归一化）**：训练/推理执行 `(X-mean)/std` 归一化，预测后 `Y_pred*std+mean` 反归一化 |
| `--parallel` | False | 启用并行计算（内存保护自动降级） |
| `--n_workers` | 4 | 并行 worker 数（最大 4） |
| `--use_gpu` | False | `torch.cuda` 可用时，推理使用 GPU |
| `--dashboard` | False | 运行后启动可视化 |
| `--skip_run` | False | 仅启动仪表盘，跳过实验 |

> **RevIN vs 标准训练**：关闭 `--revin` 时模型在原始物理尺度上训练；开启 `--revin` 时在标准化空间训练，预测后反归一化。RevIN 模式下的指标量级与 DLinear / NLinear 等深度学习基线对齐。

### 模型特定参数

```bash
# PatternSearch
python run.py --model PatternSearch --top_k 5 --weighted True

# LSHSearch
python run.py --model LSHSearch --n_hash_funcs 16 --n_tables 4 --candidate_cap_total 1024

# SAXSearch
python run.py --model SAXSearch --word_size 8 --alphabet_size 8 --bucket_top_k 8
```

## 项目结构

```
.
├── run.py                 # 统一入口（含 RevIN 预处理、内存保护调度器、TSLib Y截断）
├── models/
│   ├── PatternSearch.py   # KD-Tree KNN（float32 + torch.cdist + gc）
│   ├── LSHSearch.py       # 局部敏感哈希（uint64 打包 + 两阶段重排 + Shape广播修复）
│   └── SAXSearch.py        # 符号聚合近似（NearestNeighbors 模糊匹配 + Shape广播修复）
├── data_provider/
│   └── data_loader.py     # 数据加载（TSLib 固定边界 + float32 + 4值返回）
├── dashboard/
│   └── app.py             # Streamlit 可视化（RevIN开关、Target列绘图、expander表单、实时Log+自动刷新）
├── utils/
│   └── metrics.py         # 评估指标（MAE/MSE/RMSE/MAPE/CORR/RSE，全 float() 包裹）
└── results/              # 实验输出
    ├── experiment_log.json     # 日志（metrics + preview['preds/trues/history']）
    ├── *_preds.npy            # 完整预测结果（float32）
    └── *_trues.npy            # 完整真实值（float32）
```

## RevIN（可逆实例归一化）

RevIN 是 DLinear / NLinear 等 SOTA 深度学习模型的标配预处理，通过实例级归一化消除序列内均值/方差偏移。

### 公式

```
训练阶段（--revin）：
  X_mean = mean(X, axis=1, keepdims=True)
  X_std  = sqrt(var(X, axis=1)) + 1e-8
  X_norm = (X - X_mean) / X_std        ← 归一化输入
  Y_norm = (Y - X_mean) / X_std        ← Y 也用 X 的统计量归一化
  model.fit(X_norm, Y_norm)

推理阶段：
  X_test_norm = (X_test - X_test_mean) / X_test_std
  Y_pred_norm = model.predict(X_test_norm)
  Y_pred = Y_pred_norm * X_test_std + X_test_mean   ← 反归一化
```

### 为什么用 RevIN

| 维度 | 无归一化 | Mean-Shift | RevIN（完整） |
|------|---------|------------|--------------|
| 均值对齐 | ❌ | ✅ | ✅ |
| 方差对齐 | ❌ | ❌ | ✅ |
| TSLib 指标量级 | ❌ | 部分对齐 | ✅ |
| 深度学习 SOTA 对齐 | ❌ | ❌ | ✅ |
| 计算开销 | 无 | 极低 | 极低 |

## 算法对比

| 模型 | 搜索精度 | 检索速度 | 特点 |
|------|---------|---------|------|
| PatternSearch | 精确 | O(log n) | 欧氏距离，逆距离加权，GPU 加速 |
| LSHSearch | 近似 | O(1) | 随机投影，uint64 打包，两阶段重排 |
| SAXSearch | 模糊 | O(n) | PAA 降维，NearestNeighbors 模糊匹配 |

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
| **双尺度切换（v2.8 新增）** | `st.checkbox` 切换「原始物理尺度」/「归一化尺度」；归一化时以 history 为基准动态计算均值/标准差 |
| **波形量纲对齐（v2.8 新增）** | preview trues 改用归一化值，history/trues/preds 三者量纲统一，图表左右连贯 |
| **彻底解决缓存脏读（v2.8 新增）** | 移除 `@st.cache_data`；`subprocess` 完成后 `st.cache_data.clear()` + `st.rerun()` |
| **特征维度选择器（v2.7 新增）** | 动态下拉框，支持 ETT 预定义名称（HUFL/HULL/MUFL/.../OT）；末列标注 (Target) |
| **HTML flex 单行指标** | `display: flex` 替代 `st.columns()`，跨屏幕绝对单行 |
| **Plotly zeroline** | 所有图表 Y 轴 `zeroline=True, zerolinecolor='lightgray'` |
| **RevIN 开关** | 表单中勾选"启用 RevIN"，自动透传 `--revin` 参数 |
| **指标直接读取** | 直接从 JSON metrics 读取，绝不重新计算 |
| **实时 Log** | `subprocess.Popen` + `iter(stdout.readline)` 实时打屏，不阻塞 UI |
| **自动刷新** | 实验结束后 `st.rerun()` 自动刷新加载最新 JSON |
| **Sample ID 动态范围** | 从 `preview['count']` 动态读取最大样本数 |
| **历史连贯波形** | X 轴从 `-seq_len` 到 `pred_len-1`，x=0 分隔虚线 |
| **expander 表单** | 侧边栏表单折叠 `expanded=False`，节省主视图空间 |

## 输出

- `results/experiment_log.json` - 实验日志（metrics + preview['preds/trues/history']）
- `results/*_preds.npy` - 完整预测结果（float32）
- `results/*_trues.npy` - 完整真实值（float32）
- `results/*_X_test.npy` - 测试集输入（用于 Dashboard 连贯波形）

运行 `--dashboard` 后访问 `http://localhost:8501` 查看可视化。

## 核心模块复用

```python
from run import run_single_experiment, ExperimentRunner

# 单独运行一个实验（支持 --revin）
result = run_single_experiment({
    'model_name': 'PatternSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5,
    'mean_shift': True,  # 兼容旧 key；新版建议用 'mean_shift': True 等效 --revin
})

# RevIN 模式
result = run_single_experiment({
    'model_name': 'PatternSearch',
    'seq_len': 96,
    'pred_len': 96,
    'mean_shift': True,   # v2.6 中此 key 即 --revin
})

# 批量运行
runner = ExperimentRunner(args)
runner.run()
```
