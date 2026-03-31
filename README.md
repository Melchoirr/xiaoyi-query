# 时序预测基线系统 (v4.3 大道至简重构版)

七种检索算法，基于记忆库 / 向量索引 / 深度学习的时序预测框架。

| 模型 | 检索精度 | 核心方法 |
|------|---------|---------|
| PatternSearch | 精确 | 欧氏距离 KNN，GPU 加速 |
| LSHSearch | 近似 | 局部敏感哈希，uint64 打包 |
| SAXSearch | 模糊 | PAA 降维 + NearestNeighbors |
| DTWSearch | 弹性对齐 | GPU Sakoe-Chiba 累积 DP |
| MatrixProfileSearch | 精确子序列 | **GPU 向量化 Z-Norm cdist** |
| TS2VecSearch | 深度表示 | Dilated CNN 对比学习 + faiss |
| RAGSearch | 端到端 | Siamese Cross-Attention |

## 核心变更 (v4.3 指标计算修复 & 顶会级多通道网格图)

本次更新修复了指标计算的尺度错位 Bug，并重写了顶会级多通道对比网格图：

1. **Bug Fix 1 - 指标计算空间错位**：修复 `metrics` 在归一化空间计算的问题，改为先执行 `inverse_revin` 回到 TSLib 空间再计算 Metric，确保 MAE/MSE 反映真实物理尺度
2. **Bug Fix 2 - X_test 量纲漏失**：修复 `X_test.npy` 未执行物理尺度恢复的问题，新增 `test_set.inverse_transform` 确保所有保存的 npy 文件均为物理尺度
3. **功能增强 - 顶会级多通道网格图**：重写 `plot_cross_model_comparison()`，生成 `n_features × n_samples` 的大网格图，布局为 `figsize=(6*n_cols, 2.5*n_features)`，统一保存为 `paper_level_comparison.png`
4. **布局优化**：历史输入 `#aaaaaa`/linewidth=0.8/alpha=0.7，真实值 `#333333`/linewidth=0.5，各模型默认颜色循环/linewidth=0.3/alpha=0.85，`x=0` 处 `#cccccc` 虚线分割，dpi=300

## v4.0/v4.3 架构设计

```bash
# 核心依赖
pip install numpy pandas scikit-learn scipy rich torch psutil tqdm matplotlib pandas

# 可选依赖
pip install tslearn stumpy faiss-cpu   # 进一步加速对应模型
```

## 快速使用

```bash
# 完整参数扫描（Shell 脚本智能调度）
./scripts/run_experiments.sh --models all --seq-lens 96 --pred-lens 96 --revin-types dual --parallel --use-gpu

# 快速验证安装（单次运行）
python run.py --model PatternSearch --seq_len 96 --pred_len 96 --run_dir ./results/run_001

# 自定义实验
python run.py --model PatternSearch --seq_len 96 --pred_len 96 --top_k 5 --weighted true --run_dir ./results/run_001
```

## v4.0 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Shell 调度层 (run_experiments.sh)            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  case 语句独立循环：                                          │  │
│  │    - PatternSearch: top_k × weighted                        │  │
│  │    - DTWSearch: top_k × dtw_radius                          │  │
│  │    - RAGSearch: d_model × n_heads × epochs                  │  │
│  │  后台任务 (&) + wait 并发控制                                 │  │
│  │  实验结束：自动调用 Python 生成超级矩阵图 + 跨模型对比图        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Python 执行层 (run.py)                    │  │
│  │  单进单出：每次调用执行单一实验                               │  │
│  │  - 训练 → 推理 → 计算指标 → 保存结果                        │  │
│  │  - 追加到 summary_metrics.csv                               │  │
│  │  - 检索模型：额外保存 retrieval_meta.npz (物理尺度)          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  可视化层 (plotting.py)                      │  │
│  │  - 实验结束：自动绘制单实验波形对比图 (DPI=300)               │  │
│  │  - Shell 结束：绘制超级对比矩阵 + 溯源图                     │  │
│  │  - 自动生成前 3 个样本的跨模型对比图                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 核心模块复用

```python
# 单次实验（Python API）
from run import run_single_experiment

result = run_single_experiment({
    'model_name': 'PatternSearch',
    'seq_len': 96,
    'pred_len': 48,
    'top_k': 5,
    'weighted': True,
    'revin_type': 'dual',
    'run_dir': './results/run_001',
})
print(result['metrics'])

# 绘图工具
from plotting import (
    plot_comparison_samples,
    plot_summary_bar,
    plot_super_comparison_matrix,
    plot_retrieval_fading,
    plot_cross_model_comparison,
    generate_all_plots
)

# 生成所有分析图表
generate_all_plots('./results/run_001')

# 单独绘制超级对比矩阵
plot_super_comparison_matrix('./results/run_001', metric='MAE')

# 绘制溯源图
plot_retrieval_fading('./results/run_001/exp_id', n_samples=5)

# 绘制跨模型对比图（指定样本和特征）
plot_cross_model_comparison('./results/run_001', sample_id=0, feat_idx=-1)
```

## 算法对比

| 模型 | 精度 | 复杂度 | 关键特性 | 内存代价 |
|------|------|--------|----------|----------|
| PatternSearch | 精确 | O(n) | torch.cdist GPU 加速，逆距离加权 | ~32 MB |
| LSHSearch | 近似 | O(1) | 随机投影哈希，两阶段重排 | ~1 MB |
| SAXSearch | 模糊 | O(n) | PAA 降维，模糊匹配 | ~67 KB |
| DTWSearch | 弹性对齐 | O(n·m·r) | GPU Sakoe-Chiba 累积 DP | ~410 MB |
| **MatrixProfileSearch** | **精确子序列** | **O(n·m)** | **GPU Z-Norm 2D cdist** | ~32 MB |
| TS2VecSearch | 深度表示 | O(n) | TCN 对比学习，DataLoader workers=8 | ~100 MB |
| RAGSearch | 端到端 | O(n) | Siamese Cross-Attention | ~1.5 GB |

## 输出结构

```
results/
└── run_20260330_120000/          # Shell 脚本生成，所有实验共享
    ├── summary_metrics.csv        # 所有实验汇总
    ├── super_comparison_matrix.png # 顶会级对比热力图
    ├── model_ranking_bar.png      # 模型排名柱状图
    ├── paper_level_comparison.png  # 顶会级多通道网格图（v4.3）
    ├── cross_model_comparison_sample0.png  # 跨模型对比图（样本0）
    ├── cross_model_comparison_sample1.png  # 跨模型对比图（样本1）
    ├── cross_model_comparison_sample2.png  # 跨模型对比图（样本2）
    ├── logs/
    │   └── *.log                  # 各实验日志
    └── ETTm1_PatternSearch_seq96_pred96_k5_Rd/
        ├── params.json
        ├── metrics.json
        ├── preds.npy              # 预测值（原始物理尺度）
        ├── trues.npy
        ├── X_test.npy
        ├── retrieval_meta.npz     # 溯源证据（物理尺度，仅检索模型）
        ├── visualization.png      # 单实验波形对比图
        └── retrieval_analysis.png  # 溯源分析图（仅检索模型）
```

## Shell 脚本调度示例

```bash
# 完整参数扫描（7 个模型 × 多个超参组合）
./scripts/run_experiments.sh --models all --seq-lens 96,192 --pred-lens 96,192 --revin-types dual --parallel --use-gpu

# 仅扫描 PatternSearch
./scripts/run_experiments.sh --model PatternSearch --seq-lens 96,192 --pred-lens 96 --top_k 3,5,10 --parallel

# 仅扫描 RAGSearch（无 top_k 循环）
./scripts/run_experiments.sh --model RAGSearch --seq-lens 96,192,336 --pred-lens 96,192 --rag_d_model 128,256 --rag_epochs 50,100 --parallel

# 干跑测试（不执行，只打印命令）
./scripts/run_experiments.sh --models all --dry-run

# 限制并发数
./scripts/run_experiments.sh --models all --max-jobs 4 --parallel
```

## 版本历史

| 版本 | 更新内容 |
|------|---------|
| **v4.3** | 修复指标计算尺度错位（Metrics 须在 TSLib 空间计算）；修复 X_test.npy 漏失物理尺度恢复；重写顶会级多通道网格图（n_features × n_samples，figsize=(6*n_cols, 2.5*n_features)，统一保存 paper_level_comparison.png） |
| **v4.2** | 优化绘图视觉：自动计算画布尺寸，增粗线条(D=2.0,GT=2.0)，固定DPI=300；DTWSearch GPU OOM重试机制：`_gpu_retry_with_sleep`自动排队等待，最多重试30次渐进等待 |
| **v4.1** | 修复 Shell `--models all` Bug；修复 RevIN Feature 维度广播崩溃（axis=(1,2)）；修复溯源元数据量纲错位（反归一化回物理尺度）；新增跨模型对比图；去雾化（移除 marker，DPI=300） |
| **v4.0** | 大道至简重构：删除 Python 参数网格，Shell 全权调度；新增溯源证据输出和顶会级可视化 |
| **v3.5** | chunk 爆炸式提升，Shell 脚本智能路由 |
| **v3.4** | MatrixProfileSearch GPU Z-Norm 安全版 |
| **v3.2** | DTWSearch GPU Sakoe-Chiba；RAGSearch Siamese Cross-Attention |
| **v3.1** | 新增 DTWSearch、MatrixProfileSearch、TS2VecSearch、RAGSearch |
| **v3.0** | Dual-Dimension RevIN |
