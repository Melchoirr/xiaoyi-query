# 时序预测基线模型

基于记忆检索的时序预测，包含三种算法：**PatternSearch** (KNN)、**LSHSearch** (局部敏感哈希)、**SAXSearch** (符号聚合近似)。

## 安装

```bash
pip install numpy pandas scikit-learn scipy rich streamlit plotly torch psutil tqdm
```

## 重要更新 (v2.4 学术规范重构)

v2.4 对齐 TSLib 指标尺度 + 容错与交互体验：

| 维度 | 变更 |
|------|------|
| **指标计算空间** | Metrics 先在归一化空间计算（对齐 TSLib 0.3/0.4 量级），再 inverse_transform 用于落盘/可视化 |
| **Mean-Shift 广播安全** | 训练/预测两阶段均强制 `reshape to 3D` + `keepdims=True`，彻底消除 `could not broadcast` 报错 |
| **DLinear Instance Norm** | 训练：Y -= X_mean；预测：Y_pred += X_test_mean；指标在归一化空间计算 |
| **模型专属参数表单** | Dashboard 侧边栏根据所选模型动态显示专属参数并拼接为完整命令行 |
| **实时终端 Log** | `subprocess.Popen` + `iter(process.stdout.readline)` + `st.code()` 容器，实时滚动显示训练进度 |
| **数据切分** | 废除 ratio 比例，改为 TSLib 固定边界（月/小时时间戳） |
| **Dataloader** | `__getitem__` 返回 4 值含时间特征编码 |
| **评估指标** | MAPE/MSPE 移除 `*100`；新增 RSE、CORR；全部 `float()` 包裹防 JSON 序列化 |
| **MAPE/MSPE 鲁棒性** | Mask 机制过滤 `|true| < 1e-3` 极小值点 |
| **LSHSearch** | uint64 哈希打包 + Hamming 半径探针掩码 + 两阶段候选重排 |
| **SAXSearch** | 整数打包符号 + `sklearn.neighbors.NearestNeighbors` 替换编辑距离 |
| **新超参** | `--candidate_cap_per_table`、`--candidate_cap_total`、`--lsh_weighted`、`--bucket_top_k`、`--sax_weighted` |
| **历史上下文落盘** | `run.py` 同时保存 `*_X_test.npy`，Dashboard 显示历史波形语境 |
| **Dashboard** | `use_container_width` → `width="stretch"`；侧边栏一键启动实验面板；历史+未来连贯波形图 |

## 安装

```bash
pip install numpy pandas scikit-learn scipy rich streamlit plotly torch psutil tqdm
```

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
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | PatternSearch | 模型: PatternSearch / LSHSearch / SAXSearch / all |
| `--seq_len` | 96 | 输入序列长度 |
| `--pred_len` | 48 | 预测序列长度 |
| `--features` | M | M=多变量, S=单变量 |
| `--parallel` | False | 启用并行计算（内存保护自动降级） |
| `--n_workers` | 4 | 并行 worker 数（最大 4） |
| `--use_gpu` | False | `torch.cuda` 可用时，推理使用 GPU |
| `--dashboard` | False | 运行后启动可视化 |

说明：`--parallel` 与 `--use_gpu` 同时开启时，多进程可能争用同一块 GPU，建议大实验单进程 `--use_gpu` 或减小 `n_workers`。

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
├── run.py                 # 统一入口（含内存保护调度器）
├── models/
│   ├── PatternSearch.py   # KD-Tree KNN（float32 + gc）
│   ├── LSHSearch.py        # 局部敏感哈希（uint64 打包 + 两阶段重排）
│   └── SAXSearch.py        # 符号聚合近似（NearestNeighbors 模糊匹配）
├── data_provider/
│   └── data_loader.py     # 数据加载（TSLib 固定边界 + float32 + 4值返回）
├── dashboard/
│   └── app.py             # Streamlit 可视化（实时 Log + 历史波形）
└── results/              # 实验输出
    ├── experiment_log.json   # 日志（仅标量 metrics + 前100条预览）
    └── *_preds.npy         # 完整预测结果（float32）
```

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

所有 numpy 数组强制使用 `np.float32`，相比默认 `float64` 节省 50% 基础内存：

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

## 输出

- `results/experiment_log.json` - 实验日志（仅标量 metrics + 前 100 条预览）
- `results/*_preds.npy` - 完整预测结果（float32）
- `results/*_trues.npy` - 完整真实值（float32）

运行 `--dashboard` 后访问 `http://localhost:8501` 查看可视化。

## 核心模块复用

```python
from run import run_single_experiment, ExperimentRunner

# 单独运行一个实验
result = run_single_experiment({
    'model_name': 'PatternSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5
})

# 批量运行
runner = ExperimentRunner(args)
runner.run()
```
