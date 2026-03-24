# 时序预测基线模型

基于记忆检索的时序预测，包含三种算法：**PatternSearch** (KNN)、**LSHSearch** (局部敏感哈希)、**SAXSearch** (符号聚合近似)。

## 安装

```bash
pip install numpy pandas scikit-learn scipy rich streamlit plotly torch psutil tqdm
```

## 重要更新 (v2.1 Bug 修复)

v2.1 修复了以下关键问题：

| 问题 | 修复 |
|------|------|
| `TypeError: PatternSearch.__init__() got an unexpected keyword argument 'top_k'` | `__init__` 参数名改为 `top_k`，所有模型末尾加 `**kwargs` |
| `ValueError: non-broadcastable output operand...shape (662064,1) doesn't match...shape (662064,7)` | 移除 `Y_pred[:,:,0]` 破坏性切片；哈希桶 sum_cache 形状改为 `(pred_len, n_features)` |
| 单模型失败导致整个脚本崩溃 | `run_single_experiment` 异常隔离，单个失败继续执行下一个 |
| 实验失败后仪表盘未启动 | `--dashboard` 参数无论实验结果如何必定启动 |
| 终端无进度条 | 引入 `logging` + `tqdm`，带时间戳和实验进度 |

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
| `--features` | M | M=多变量, S=单变量（内存敏感场景建议 S） |
| `--parallel` | False | 启用并行计算（内存保护自动降级） |
| `--n_workers` | 4 | 并行 worker 数（最大 4） |
| `--use_gpu` | False | `torch.cuda` 可用时，推理使用 GPU（PatternSearch/LSH/SAX） |

说明：`--parallel` 与 `--use_gpu` 同时开启时，多进程可能争用同一块 GPU，建议大实验单进程 `--use_gpu` 或减小 `n_workers`。
| `--dashboard` | False | 运行后启动可视化 |

### 模型特定参数

```bash
# PatternSearch
python run.py --model PatternSearch --top_k 5 --weighted True

# LSHSearch
python run.py --model LSHSearch --n_hash_funcs 16 --n_tables 4

# SAXSearch
python run.py --model SAXSearch --word_size 8 --alphabet_size 8
```

## 项目结构

```
.
├── run.py                 # 统一入口（含内存保护调度器）
├── models/
│   ├── PatternSearch.py   # KD-Tree KNN（float32 + gc）
│   ├── LSHSearch.py      # 局部敏感哈希（哈希桶存均值）
│   └── SAXSearch.py      # 符号聚合近似（预聚合压缩）
├── data_provider/
│   └── data_loader.py     # 数据加载（float32 + 预分配）
├── dashboard/
│   └── app.py            # Streamlit 可视化
└── results/               # 实验输出
    ├── experiment_log.json  # 日志（仅标量 metrics + 前100条预览）
    └── *_preds.npy         # 完整预测结果（float32）
```

## 算法对比

| 模型 | 搜索精度 | 检索速度 | 特点 |
|------|---------|---------|------|
| PatternSearch | 精确 | O(log n) | 欧氏距离，逆距离加权 |
| LSHSearch | 近似 | O(1) | 随机投影，哈希碰撞 |
| SAXSearch | 模糊 | O(n) | PAA降维，编辑距离 |

## 内存优化（v2.0）

本项目针对 8核32G 环境下的 OOM 问题进行了系统性优化，理论上可将峰值内存从 30GB+ 压制到 **4~8GB**。

### 优化措施概览

| 层级 | 文件 | 优化手段 | 预期内存收益 |
|------|------|---------|------------|
| 数据加载 | `data_provider/data_loader.py` | float64→float32，预分配，del+gc | ~50% 降幅 |
| 实验调度 | `run.py` | 实验间强制 gc.collect()，JSON 截断100条 | ~1~2GB |
| 并行保护 | `run.py` | 内存>85%回退串行，max_workers=4 | 避免峰值叠加 |
| SAX 索引 | `models/SAXSearch.py` | 哈希桶存均值而非索引列表 | 10~100x 压缩 |
| LSH 索引 | `models/LSHSearch.py` | 哈希桶存均值而非索引列表 | 10~100x 压缩 |
| NPY 落盘 | `run.py` | 完整预测只存.npy，不进JSON | JSON 体积从 MB→KB |

### 1. 数据加载（float32 + 预分配）

所有 numpy 数组强制使用 `np.float32`，相比默认 `float64` 节省 50% 基础内存：

```python
# data_loader.py
DTYPE = np.float32

# CSV 读取后立即转换为 float32，避免后续 dtype 转换
raw = df_data[cols_data].values.astype(DTYPE)

# 预分配而非 append 列表再转 np.array
X_all = np.empty((n_samples, seq_len, n_feature), dtype=DTYPE)
for i in range(n_samples):
    X_all[i] = dataset[i][0]
```

### 2. 哈希桶预聚合（SAX / LSH）

**原来：** 哈希桶存 `List[int]`（每个样本索引），一个桶有 1000 条样本时存 1000 个 int（4~8KB）。

**现在：** 哈希桶存 `(mean_Y, count)` 元组，每个桶只存 1 个 float32 数组（`pred_len * 4 bytes`），压缩比可达 10~100x：

```python
# models/SAXSearch.py / LSHSearch.py
# fit() 中在线累加
self.sax_dict: Dict[str, Tuple[np.ndarray, int]] = {}
for i, sax_str in enumerate(sax_strings):
    if sax_str not in sum_cache:
        sum_cache[sax_str] = np.zeros(y_dim, dtype=np.float32)
        count_cache[sax_str] = 0
    sum_cache[sax_str] += Y_flat[i]
    count_cache[sax_str] += 1

# 最终每个桶只存均值
for sax_str in sum_cache:
    cnt = count_cache[sax_str]
    mean_Y = (sum_cache[sax_str] / cnt).astype(np.float32)
    self.sax_dict[sax_str] = (mean_Y, cnt)
```

### 3. 实验调度 GC + JSON 截断

```python
# run.py - 每次实验后强制 GC
result = run_single_experiment(cfg)
self.results.append(result)
del result   # 删除引用
gc.collect()  # 触发垃圾回收

# JSON 中只保留前 100 条预览，完整数据落盘 .npy
MAX_PREVIEW = 100
preview_pred = Y_pred_orig[:MAX_PREVIEW].astype(np.float32).tolist()
```

### 4. 并行内存保护

```python
# run.py - 动态内存检测
MEMORY_THRESHOLD = 0.85  # 超过 85% 则回退串行

def _memory_check(self) -> bool:
    mem = psutil.virtual_memory()
    if mem.percent / 100.0 >= self.MEMORY_THRESHOLD:
        print(f"[警告] 内存占用 {mem.percent:.1%} >= {self.MEMORY_THRESHOLD:.1%}，回退串行")
        return False
    return True

# 并行 worker 数限制为 4（而非 CPU 核数）
with ProcessPoolExecutor(max_workers=min(n_workers, total, 4)) as executor:
    ...
```

## 内存估算参考

以 ETTm1 数据集，`seq_len=96, pred_len=48, features=M`（7 特征）为例：

| 数据结构 | float64 原始 | float32 优化后 |
|---------|-------------|---------------|
| 训练集 X [~26K, 96, 7] | ~55 MB | ~28 MB |
| 训练集 Y [~26K, 48, 7] | ~27 MB | ~14 MB |
| SAX 索引（原：索引列表） | ~80 MB | ~2 MB |
| LSH 索引（原：索引列表） | ~80 MB | ~2 MB |
| PatternSearch memory | ~55 MB | ~28 MB |
| 测试集 X/Y 峰值 | ~15 MB | ~8 MB |
| **单模型峰值合计** | **~350 MB** | **~100 MB** |
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
