# 时序预测统一框架
# Time Series Forecasting Unified Framework

"""
支持9种时序预测模型：

检索模型（7种）:
  - PatternSearch: 欧氏距离 KNN 检索
  - LSHSearch: 局部敏感哈希检索
  - SAXSearch: PAA+SAX 符号化检索
  - DTWSearch: 动态时间规整检索
  - MatrixProfileSearch: GPU Z-Norm 子序列检索
  - TS2VecSearch: 深度对比学习检索
  - RAGSearch: Siamese Cross-Attention RAG

深度学习模型（2种）:
  - DLinear: 季节性/趋势分解 + 线性预测
  - PatchTST: Patch + Transformer 编码器

使用示例:
  # 检索模型
  python run.py --model PatternSearch --data ETTh1 --seq_len 512 --pred_len 96

  # 深度学习模型
  python run.py --model DLinear --data ETTh1 --seq_len 512 --pred_len 96 --is_training 1

  # 模型融合
  python run.py --mode fusion --fusion_models PatternSearch,DLinear
"""

__version__ = "5.0.0"
