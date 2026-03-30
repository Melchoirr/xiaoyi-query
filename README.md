# 时序预测基线系统 (v3.5)

七种检索算法，基于记忆库 / 向量索引 / 深度学习的时序预测框架。

| 模型 | 检索精度 | 核心方法 |
|------|---------|---------|
| PatternSearch | 精确 | 欧氏距离 KNN，GPU 加速 |
| LSHSearch | 近似 | 局部敏感哈希，uint64 打包 |
| SAXSearch | 模糊 | PAA 降维 + NearestNeighbors |
| DTWSearch | 弹性对齐 | GPU Sakoe-Chiba 累积 DP（v3.2） |
| MatrixProfileSearch | 精确子序列 | **GPU 向量化 Z-Norm cdist**（v3.4） |
| TS2VecSearch | 深度表示 | Dilated CNN 对比学习 + faiss |
| RAGSearch | 端到端 | Siamese Cross-Attention（v3.2） |

## 安装

```bash
# 核心依赖
pip install numpy pandas scikit-learn scipy rich torch psutil tqdm matplotlib pandas

# 可选依赖
pip install tslearn stumpy faiss-cpu   # 进一步加速对应模型
```

## 快速使用

```bash
# 单模型快速运行（自动生成 run_YYYYMMDD_HHMMSS/）
python run.py --model MatrixProfileSearch --seq_len 96 --pred_len 48 --top_k 5

# 参数扫描（Shell 脚本，自动时间戳隔离 + 智能路由）
bash scripts/run_experiments.sh --models all --seq-lens 96,192 --pred-lens 96 --revin-types dual --parallel --use-gpu

# 指定输出目录（覆盖自动时间戳，用于调度系统对接）
python run.py --model all --run_dir ./results/my_sweep_001

# 参数说明
# --model: PatternSearch / LSHSearch / SAXSearch / DTWSearch /
#          MatrixProfileSearch / TS2VecSearch / RAGSearch / all
# --seq_len: 96 / 192 / 336 / 720（TSLib 标准）
# --pred_len: 48 / 96 / 192 / 336 / 720
# --revin_type: none / temporal / feature / dual
# --run_dir: 输出目录（Shell 脚本统一管理，不在 Python 内生成时间戳）
```

## 各模型专属参数（v3.5 默认值）

```bash
# PatternSearch / DTWSearch / MatrixProfileSearch — chunk 爆炸式提升
python run.py --model PatternSearch     --top_k 5 --weighted True
python run.py --model DTWSearch         --top_k 5 --dtw_radius 5 --predict_chunk_size 4096
python run.py --model MatrixProfileSearch --top_k 5 --mp_chunk_size 8192

# TS2VecSearch — batch 1024, epochs 100, DataLoader num_workers=8
python run.py --model TS2VecSearch --hidden_dim 64 --ts2vec_epochs 100 --ts2vec_batch_size 1024

# RAGSearch — d_model=256, heads=8, batch=1024, epochs=100
python run.py --model RAGSearch --rag_d_model 256 --rag_n_heads 8 --rag_epochs 100
```

## 输出结构（v3.5 Run-Level 时间戳隔离）

```
results/
└── run_20260325_143052/          # Shell 脚本生成，所有实验共享
    ├── experiment_log.json
    ├── summary_metrics.csv        # 所有实验汇总
    ├── summary_MAE_bar.png       # MAE 柱状图
    ├── logs/
    │   ├── PatternSearch_seq96_pred96_k5_Rd.log
    │   └── RAGSearch_seq96_pred96_k5_Rd.log
    └── ETTm1_PatternSearch_seq96_pred96_k5_Rd/
        ├── params.json
        ├── metrics.json
        ├── preds.npy             # 预测值（原始物理尺度）
        ├── trues.npy
        ├── X_test.npy
        └── visualization.png       # 四段线对比图
```

## 算法对比

| 模型 | 精度 | 复杂度 | 关键特性 |
|------|------|--------|---------|
| PatternSearch | 精确 | O(n) | torch.cdist GPU 加速，逆距离加权，chunk=4096 |
| LSHSearch | 近似 | O(1) | 随机投影哈希，两阶段重排 |
| SAXSearch | 模糊 | O(n) | PAA 降维，模糊匹配 |
| DTWSearch | 弹性对齐 | O(chunk·n·m·r) | GPU Sakoe-Chiba 累积 DP，chunk=4096（v3.5） |
| **MatrixProfileSearch** | **精确子序列** | **O(chunk·n·m)** | **GPU Z-Norm 2D cdist，chunk=8192，V100 ~10s（v3.4）** |
| TS2VecSearch | 深度表示 | O(n) | TCN 对比学习，DataLoader workers=8，batch=1024，epochs=100 |
| RAGSearch | 端到端 | O(n) | Siamese Cross-Attention，d_model=256，batch=1024，epochs=100 |

## Shell 脚本智能路由（v3.5）

`run_experiments.sh` 采用 `case` 路由，每个模型专属参数网格：

```bash
# PatternSearch / LSHSearch / SAXSearch / DTWSearch / TS2VecSearch：
#   遍历 top_k、revin_type、seq_len、pred_len

# RAGSearch（无 top_k 循环，节约 N×k 次无意义调用）：
#   仅遍历 seq_len、pred_len、revin_type（top_k 固定为默认值）

# 示例：仅扫描 RAGSearch
bash scripts/run_experiments.sh \
    --models RAGSearch \
    --seq-lens 96,192,336 \
    --pred-lens 96,192 \
    --revin-types dual,temporal \
    --parallel --use-gpu
```

## 项目结构

```
.
├── run.py                      # 统一入口（--run_dir 支持、RevIN、实验持久化）
├── plotting.py                 # 四段线对比图 + MAE 柱状图（全英文，Linux 安全）
├── models/
│   ├── PatternSearch.py         # 欧氏 KNN（chunk=4096）
│   ├── LSHSearch.py            # LSH
│   ├── SAXSearch.py            # SAX
│   ├── DTWSearch.py            # DTW Sakoe-Chiba GPU DP（chunk=4096）
│   ├── MatrixProfileSearch.py  # GPU Z-Norm 2D cdist（chunk=8192）
│   ├── TS2VecSearch.py         # TCN 对比学习（DataLoader workers=8）
│   └── RAGSearch.py            # Siamese Cross-Attention（d_model=256）
├── data_provider/
│   └── data_loader.py          # TSLib 数据加载，float32
├── dashboard/
│   └── app.py                  # Streamlit 交互式探查
├── scripts/
│   └── run_experiments.sh      # 智能路由参数扫描（v3.5）
└── results/
    └── run_YYYYMMDD_HHMMSS/    # Run-Level 时间戳隔离（v3.5）
```

## 核心模块复用

```python
from run import run_single_experiment, ExperimentRunner

# 单独运行（自动生成 run_YYYYMMDD_HHMMSS/）
result = run_single_experiment({
    'model_name': 'MatrixProfileSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5,
    'normalize': True,
    'revin_type': 'temporal',
})
print(result['metrics'])

# 指定输出目录
runner = ExperimentRunner(args)  # args.run_dir 控制输出位置
runner.run()

# 独立绘图
from plotting import plot_comparison_samples, plot_summary_bar
plot_comparison_samples(
    history=np.load('results/{run_id}/{exp_id}/X_test.npy'),
    preds=np.load('results/{run_id}/{exp_id}/preds.npy'),
    trues=np.load('results/{run_id}/{exp_id}/trues.npy'),
    seq_len=96, pred_len=48, n_features=7,
    model_name='MatrixProfileSearch',
    save_path='results/{run_id}/{exp_id}/visualization.png',
)
plot_summary_bar('results/{run_id}/summary_metrics.csv', metric='MAE')
```

## 版本历史

| 版本 | 更新内容 |
|------|---------|
| **v3.5** | `--run_dir` 参数（Shell 统一管理时间戳）；`num_workers=8, pin_memory=True` 解放 DataLoader CPU 瓶颈；chunk 爆炸式提升（MP→8192, DTW/Pattern→4096）；batch 1024 / d_model 256 / epochs 100；Shell 脚本智能路由（RAGSearch 跳过 top_k 循环）；plotting.py 全英文学术标签 |
| **v3.4** | MatrixProfileSearch: 彻底移除 Python 循环，GPU Z-Norm 2D cdist 安全版，chunk=4096，V100 ~10 秒 |
| **v3.3** | MatrixProfileSearch: 移除 stumpy 依赖，纯 PyTorch 向量化 |
| **v3.2** | DTWSearch: GPU Sakoe-Chiba 累积 DP；MatrixProfileSearch: stumpy 工业级集成；RAGSearch: Siamese Cross-Attention |
| **v3.1** | 新增 DTWSearch、MatrixProfileSearch、TS2VecSearch、RAGSearch 共 4 个前沿模型 |
| **v3.0** | Dual-Dimension RevIN（temporal/feature/dual）、目录规范化（results/{exp_id}/）、静态可视化 |
