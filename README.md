# 时序预测基线模型

基于记忆检索的时序预测，包含三种算法：**PatternSearch** (KNN)、**LSHSearch** (局部敏感哈希)、**SAXSearch** (符号聚合近似)。

## 安装

```bash
pip install numpy pandas scikit-learn scipy rich streamlit plotly
```

## 使用方法

### 统一入口 `run.py`

```bash
# 基本用法
python run.py --model PatternSearch                          # 单模型
python run.py --model all                                     # 所有模型
python run.py --model PatternSearch,LSHSearch                 # 指定模型

# 参数网格
python run.py --model all --seq_len 96 192 --pred_len 24 48 96

# 并行 + 仪表盘
python run.py --model all --parallel --dashboard

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
| `--parallel` | False | 启用并行计算 |
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
├── run.py                 # 统一入口
├── models/
│   ├── PatternSearch.py   # KD-Tree KNN
│   ├── LSHSearch.py      # 局部敏感哈希
│   └── SAXSearch.py      # 符号聚合近似
├── data_provider/
│   └── data_loader.py     # 数据加载
├── dashboard/
│   └── app.py            # Streamlit 可视化
└── results/               # 实验输出
```

## 算法对比

| 模型 | 搜索精度 | 检索速度 | 特点 |
|------|---------|---------|------|
| PatternSearch | 精确 | O(log n) | 欧氏距离，逆距离加权 |
| LSHSearch | 近似 | O(1) | 随机投影，哈希碰撞 |
| SAXSearch | 模糊 | O(n) | PAA降维，编辑距离 |

## 输出

- `results/experiment_log.json` - 实验日志
- `results/*_preds.npy` - 预测结果
- `results/*_trues.npy` - 真实值

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
