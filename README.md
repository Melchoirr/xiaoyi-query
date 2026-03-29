# 时序预测基线系统 (v3.3)

七种检索算法，基于记忆库 / 向量索引 / 深度学习的时序预测框架。

| 模型 | 检索精度 | 核心方法 |
|------|---------|---------|
| PatternSearch | 精确 | 欧氏距离 KNN，GPU 加速 |
| LSHSearch | 近似 | 局部敏感哈希，uint64 打包 |
| SAXSearch | 模糊 | PAA 降维 + NearestNeighbors |
| DTWSearch | 弹性对齐 | GPU Sakoe-Chiba 累积 DP（v3.2） |
| MatrixProfileSearch | 精确子序列 | **GPU 向量化 Z-Norm cdist**（v3.3） |
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
# 单模型
python run.py --model MatrixProfileSearch --seq_len 96 --pred_len 48 --top_k 5

# 参数扫描（Shell 脚本）
bash scripts/run_experiments.sh --models MatrixProfileSearch --seq-lens 96,192

# RevIN 归一化（temporal / feature / dual）
python run.py --model all --revin_type dual --seq_len 96 --pred_len 48

# 参数说明
# --model: PatternSearch / LSHSearch / SAXSearch / DTWSearch /
#          MatrixProfileSearch / TS2VecSearch / RAGSearch / all
# --seq_len: 96 / 192 / 336 / 720（TSLib 标准）
# --pred_len: 48 / 96 / 192 / 336 / 720
# --revin_type: none / temporal / feature / dual
```

## 各模型专属参数

```bash
python run.py --model PatternSearch  --top_k 5 --weighted True
python run.py --model LSHSearch     --n_hash_funcs 16 --n_tables 4
python run.py --model SAXSearch     --word_size 8 --alphabet_size 8
python run.py --model DTWSearch     --top_k 5 --dtw_radius 5
python run.py --model MatrixProfileSearch --top_k 5 --normalize True
python run.py --model TS2VecSearch --hidden_dim 64 --ts2vec_epochs 10 --batch_size 128
python run.py --model RAGSearch     --rag_d_model 32 --rag_n_heads 4 --rag_epochs 10
```

## 输出结构

```
results/
├── summary_metrics.csv        # 所有实验汇总（MAE/MSE/RMSE/MAPE/RSE/CORR）
├── experiment_log.json
├── logs/                     # 每实验 {exp_id}.log
└── {exp_id}/                # 每实验独立文件夹
    ├── params.json           # 配置快照
    ├── metrics.json          # 指标
    ├── preds.npy             # 预测值（原始物理尺度）
    ├── trues.npy             # 真实值
    ├── X_test.npy            # 测试集输入
    └── visualization.png      # 四段线对比图（自动生成）
```

## 算法对比

| 模型 | 精度 | 复杂度 | 关键特性 |
|------|------|--------|---------|
| PatternSearch | 精确 | O(n) | torch.cdist GPU 加速，逆距离加权 |
| LSHSearch | 近似 | O(1) | 随机投影哈希，两阶段重排 |
| SAXSearch | 模糊 | O(n) | PAA 降维，模糊匹配 |
| DTWSearch | 弹性对齐 | O(chunk·n·m·r) | GPU Sakoe-Chiba 累积 DP，显存 ≈410 MB（v3.2） |
| **MatrixProfileSearch** | **精确子序列** | **O(chunk·n·m)** | **GPU 向量化 Z-Norm cdist，无 Python 循环，V100 约 10 秒（v3.3）** |
| TS2VecSearch | 深度表示 | O(n) | TCN 对比学习 + faiss GPU 检索 |
| RAGSearch | 端到端 | O(n) | Siamese Cross-Attention，全量 Keys 实时更新 |

## 项目结构

```
.
├── run.py                      # 统一入口（RevIN、目录规范、实验持久化）
├── plotting.py                 # 四段线对比图 + 指标柱状图
├── models/
│   ├── PatternSearch.py         # 欧氏 KNN
│   ├── LSHSearch.py            # LSH
│   ├── SAXSearch.py            # SAX
│   ├── DTWSearch.py            # DTW Sakoe-Chiba GPU DP（v3.2）
│   ├── MatrixProfileSearch.py  # GPU Z-Norm cdist 检索（v3.3）
│   ├── TS2VecSearch.py         # TCN 对比学习（v3.2）
│   └── RAGSearch.py            # Siamese Cross-Attention（v3.2）
├── data_provider/
│   └── data_loader.py          # TSLib 数据加载，float32
├── dashboard/
│   └── app.py                  # Streamlit 交互式探查
├── scripts/
│   └── run_experiments.sh      # 参数扫描 + tee 日志
└── results/                    # 实验输出
```

## 核心模块复用

```python
from run import run_single_experiment, ExperimentRunner

# 单独运行
result = run_single_experiment({
    'model_name': 'MatrixProfileSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5,
    'normalize': True,
    'revin_type': 'temporal',
})
print(result['metrics'])

# 批量运行
runner = ExperimentRunner(args)
runner.run()

# 独立绘图
from plotting import plot_comparison_samples, plot_summary_bar
plot_comparison_samples(
    history=np.load('results/{exp_id}/X_test.npy'),
    preds=np.load('results/{exp_id}/preds.npy'),
    trues=np.load('results/{exp_id}/trues.npy'),
    seq_len=96, pred_len=48, n_features=7,
    model_name='MatrixProfileSearch',
    save_path='results/{exp_id}/visualization.png',
)
plot_summary_bar('results/summary_metrics.csv', metric='MAE')
```

## 版本历史

| 版本 | 更新内容 |
|------|---------|
| **v3.3** | MatrixProfileSearch: 彻底移除 Python 循环，**全 GPU 向量化 Z-Norm torch.cdist**，多变量特征维独立计算再聚合，V100 35K 样本约 10 秒；`_HAS_STUMPY` 降为备用；所有模型 GPU 自动检测 |
| **v3.2** | DTWSearch: GPU Sakoe-Chiba 累积 DP；MatrixProfileSearch: stumpy 工业级集成；RAGSearch: Siamese Cross-Attention；README 精简 |
| **v3.1** | 新增 DTWSearch、MatrixProfileSearch、TS2VecSearch、RAGSearch 共 4 个前沿模型 |
| **v3.0** | Dual-Dimension RevIN（temporal/feature/dual）、目录规范化（results/{exp_id}/）、静态可视化 |
