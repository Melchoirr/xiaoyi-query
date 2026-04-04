# 时序预测统一框架 (Time Series Forecasting Unified Framework)

## 项目简介

本项目是一个统一的时序预测框架，基于 Example 项目架构，支持 **9 种时序预测模型**：

### 检索模型（7种）

| 模型 | 精度 | 核心方法 |
|------|------|----------|
| PatternSearch | 精确 | 欧氏距离 KNN，GPU 加速 |
| LSHSearch | 近似 | 局部敏感哈希，uint64 打包 |
| SAXSearch | 模糊 | PAA 降维 + NearestNeighbors |
| DTWSearch | 弹性对齐 | GPU Sakoe-Chiba 累积 DP |
| MatrixProfileSearch | 精确子序列 | GPU Z-Norm 2D cdist |
| TS2VecSearch | 深度表示 | Dilated CNN 对比学习 + faiss |
| RAGSearch | 端到端 | Siamese Cross-Attention |

### 深度学习模型（2种）

| 模型 | 核心方法 |
|------|----------|
| DLinear | 季节性/趋势分解 + 线性预测 |
| PatchTST | Patch + Transformer 编码器 |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 检索模型使用

```bash
# PatternSearch (欧氏距离 KNN)
python run.py --model PatternSearch --data ETTh1 --seq_len 96 --pred_len 96

# LSHSearch (局部敏感哈希)
python run.py --model LSHSearch --data ETTh1 --seq_len 96 --pred_len 96

# SAXSearch (PAA+SAX 符号化)
python run.py --model SAXSearch --data ETTh1 --seq_len 96 --pred_len 96

# DTWSearch (动态时间规整)
python run.py --model DTWSearch --data ETTh1 --seq_len 96 --pred_len 96

# MatrixProfileSearch (GPU Z-Norm)
python run.py --model MatrixProfileSearch --data ETTh1 --seq_len 96 --pred_len 96

# TS2VecSearch (深度对比学习)
python run.py --model TS2VecSearch --data ETTh1 --seq_len 96 --pred_len 96

# RAGSearch (Siamese Cross-Attention)
python run.py --model RAGSearch --data ETTh1 --seq_len 96 --pred_len 96
```

### 深度学习模型使用

```bash
# DLinear
python run.py --model DLinear --data ETTh1 --seq_len 96 --pred_len 96 --is_training 1

# PatchTST
python run.py --model PatchTST --data ETTh1 --seq_len 96 --pred_len 96 --is_training 1
```

### 模型融合

```bash
# XGBoost Stacking 融合
python run.py --mode fusion --fusion_models PatternSearch,DLinear
```

## 检索模型参数说明

### PatternSearch
```bash
python run.py --model PatternSearch \
  --top_k 5 \                    # Top-K 近邻数量
  --weighted true \              # 是否使用逆距离加权
  --predict_chunk_size 4096      # 预测分块大小
```

### LSHSearch
```bash
python run.py --model LSHSearch \
  --n_hash_funcs 16 \            # 哈希函数数量
  --n_tables 4 \                 # 哈希表数量
  --hamming_radius 2 \           # 汉明半径
  --candidate_cap_per_table 256 \ # 每表候选上限
  --candidate_cap_total 1024      # 总候选上限
```

### SAXSearch
```bash
python run.py --model SAXSearch \
  --word_size 8 \               # 词大小
  --alphabet_size 8 \           # 字母表大小
  --epsilon_threshold 1.0 \      # 距离阈值
  --bucket_top_k 8               # Bucket Top-K
```

### DTWSearch
```bash
python run.py --model DTWSearch \
  --top_k 5 \                   # Top-K
  --dtw_radius 5 \             # Sakoe-Chiba 半径
  --predict_chunk_size 128      # 预测分块大小
```

### MatrixProfileSearch
```bash
python run.py --model MatrixProfileSearch \
  --top_k 5 \                   # Top-K
  --mp_normalize true \          # 是否 Z-Norm
  --predict_chunk_size 96 \    # 预测分块大小
  --train_chunk_size 1024       # 训练分块大小
```

### TS2VecSearch
```bash
python run.py --model TS2VecSearch \
  --hidden_dim 64 \            # 隐向量维度
  --epochs 10 \                 # 训练轮数
  --batch_size 128 \            # 批大小
  --top_k 5                     # Top-K
```

### RAGSearch
```bash
python run.py --model RAGSearch \
  --rag_d_model 32 \           # 隐向量维度
  --rag_n_heads 4 \             # 注意力头数
  --rag_epochs 10 \             # 训练轮数
  --rag_batch_size 128          # 批大小
```

## 项目结构

```
.
├── models/                     # 模型目录
│   ├── __init__.py             # 模型注册表（包装器）
│   ├── PatternSearch.py        # 欧氏距离 KNN 检索
│   ├── LSHSearch.py            # 局部敏感哈希检索
│   ├── SAXSearch.py             # PAA+SAX 符号化检索
│   ├── DTWSearch.py             # 动态时间规整检索
│   ├── MatrixProfileSearch.py   # GPU Z-Norm 子序列检索
│   ├── TS2VecSearch.py          # 深度对比学习检索
│   └── RAGSearch.py             # Siamese Cross-Attention RAG
│
├── layers/                     # 神经网络层
│   ├── RevIN.py               # Reversible Instance Normalization
│   └── Embed.py               # PatchEmbedding + PositionalEncoding
│
├── exp/                        # 实验模块
│   ├── exp_basic.py            # 基础实验类
│   └── exp_long_term_forecasting.py  # 长时预测实验类
│
├── data_provider/              # 数据提供模块
│   ├── data_factory.py         # 数据工厂
│   └── data_loader.py          # ETT 数据集加载器
│
├── utils/                      # 工具模块
│   ├── metrics.py             # 评估指标 (MAE/MSE/RMSE/MAPE/RSE/CORR)
│   ├── timefeatures.py         # 时间特征提取
│   └── tools.py               # EarlyStopping、学习率调整
│
├── fusion/                     # 模型融合模块
│   └── stacking.py            # XGBoost Stacking 融合
│
├── run.py                      # 主入口
├── requirements.txt            # 依赖
└── README.md                  # 本文档
```

## 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│                         run.py (统一入口)                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐              ┌────────────────┐           │
│  │  检索模型路径   │              │  深度学习路径    │           │
│  │  is_retrieval=True│            │  is_retrieval=False│         │
│  ├────────────────┤              ├────────────────┤           │
│  │ PatternSearch  │              │ DLinear        │           │
│  │ LSHSearch      │              │ PatchTST       │           │
│  │ SAXSearch      │              │                │           │
│  │ DTWSearch      │              │                │           │
│  │ MatrixProfile  │              │                │           │
│  │ TS2VecSearch   │              │                │           │
│  │ RAGSearch      │              │                │           │
│  └───────┬────────┘              └───────┬────────┘           │
│          │                               │                     │
│          ▼                               ▼                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              exp/exp_long_term_forecasting.py              │ │
│  │  - train(): 构建记忆库 / 梯度训练                           │ │
│  │  - test(): 批量推理                                        │ │
│  │  - vali(): 验证                                            │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  data_provider/data_loader.py               │ │
│  │  - Dataset_ETT_hour / Dataset_ETT_minute                  │ │
│  │  - StandardScaler 归一化                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 数据集

支持以下数据集：

| 数据集 | 频率 | 特征数 |
|--------|------|--------|
| ETTh1 | 小时 | 7 |
| ETTh2 | 小时 | 7 |
| ETTm1 | 分钟 | 7 |
| ETTm2 | 分钟 | 7 |

## 版本历史

| 版本 | 更新内容 |
|------|---------|
| **v5.0** | 统一框架重构，基于 Example 项目架构，支持 9 种模型 |
| v4.3 | 修复指标计算尺度错位问题 |
| v4.0 | 大道至简重构：删除参数网格，Shell 全权调度 |
| v3.1 | 新增 DTWSearch、MatrixProfileSearch、TS2VecSearch、RAGSearch |
| v3.0 | Dual-Dimension RevIN |
