"""
scripts/train_ranker.py
=======================
Layer 3 离线精排训练脚本 — 用回测数据训练 XGBoost 精排模型。

【训练流程】
    1. 从 ETT 数据集中随机采样 N 个时间点作为"模拟当前时刻"。
    2. 对每个采样点，获取其 history_x（历史窗口）和 true_future_y（真实未来走向）。
    3. 用 ONNXEncoder 对 history_x 编码，得到 query_vector。
    4. 用 QdrantRetriever 召回 Top-K 相似历史片段。
    5. 构造回归标签 Y_target：
         用 Pearson 相关系数衡量召回片段 future_y_k 与真实 true_future_y 的相似度，
         相关系数越高 → 该片段越有价值 → 精排模型应给予更高权重。
    6. 构造交叉特征（与线上 FusionRanker 完全对齐，见 _build_cross_features）：
         - distance         (向量距离)
         - abs(mu_q - mu_k) (均值差异)
         - abs(sigma_q - sigma_k) (标准差差异)
         - is_same_month    (月份是否相同)
         - is_same_time_of_day (时段是否相同)
    7. 用 XGBoost 学习：给定 (query, doc) 交叉特征 → 预测该 doc 的质量分数。
    8. 将训练好的 XGBoost 模型导出为 models/xgb_ranker.json。

【特征一致性保证】
    _build_cross_features() 与线上 components/ranker.py 中的同名函数逻辑完全一致，
    通过共享的 extract_time_features() 保证特征口径统一，杜绝特征穿越。

【使用方式】
    # 方式 1: 使用默认参数（5000 个采样点，Top-K=20）
    python scripts/train_ranker.py

    # 方式 2: 指定采样数量和 K
    python scripts/train_ranker.py --num-samples 10000 --top-k 20

    # 方式 3: 指定 ETT 数据集和列
    python scripts/train_ranker.py --file ETTh1 --column OT --num-samples 5000

    # 方式 4: 强制重新训练（跳过模型已存在的检查）
    python scripts/train_ranker.py --force
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from core.config import cfg
from core.processor import TSProcessor
from components.encoder import ONNXEncoder
from components.retriever import QdrantRetriever


# ---------------------------------------------------------------------------
# 时间特征提取（与 ingest_ett.py / ranker.py 保持完全一致）
# ---------------------------------------------------------------------------

def extract_time_features(ts) -> dict:
    """
    从时间戳中提取结构化时间特征。
    必须与 scripts/ingest_ett.py 中的 extract_time_features() 完全对齐。
    """
    hour = ts.hour
    is_weekend = ts.weekday() >= 5  # Saturday=5, Sunday=6

    if 0 <= hour < 6:
        time_of_day = "night"
    elif 6 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 18:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"

    return {
        "month": ts.month,
        "hour": hour,
        "is_weekend": is_weekend,
        "time_of_day": time_of_day,
    }


# ---------------------------------------------------------------------------
# 交叉特征构造（与线上 FusionRanker 完全对齐）
# ---------------------------------------------------------------------------

def _build_cross_features(
    query_metadata: dict,
    doc_payload: dict,
    distance: float,
) -> np.ndarray:
    """
    为 (query, doc) 对构造精排模型的输入特征向量。

    特征列表（共 7 维，顺序固定，与线上 FusionRanker 完全一致）：
        idx  name                     description
        0    distance                 向量距离（1 - cosine_score）
        1    abs_mu_diff              |mu_query - mu_doc|  均值差异
        2    abs_sigma_diff           |sigma_query - sigma_doc|  标准差差异
        3    is_same_month            doc.month == query.month ? 1 : 0
        4    is_same_time_of_day      doc.time_of_day == query.time_of_day ? 1 : 0
        5    is_same_weekday          doc.is_weekend == query.is_weekend ? 1 : 0
        6    mu_ratio                 mu_doc / (mu_query + 1e-6)  均值比率（对尺度差异建模）

    参数:
        query_metadata: Query 元数据字典，包含 month, hour, is_weekend, time_of_day
        doc_payload:    Qdrant 召回片段的 payload，包含 month, hour, is_weekend, time_of_day
        distance:       向量距离（1 - cosine_score）

    返回:
        numpy.ndarray，shape = (7,)，float32
    """
    q = query_metadata
    d = doc_payload

    # 统计量差异特征
    mu_q = q.get("mu", 0.0)
    mu_k = d.get("mu", 0.0)
    sigma_q = q.get("sigma", 1.0)
    sigma_k = d.get("sigma", 1.0)

    abs_mu_diff = abs(mu_q - mu_k)
    abs_sigma_diff = abs(sigma_q - sigma_k)

    # 均值比率（避免除零，加 epsilon）
    mu_ratio = mu_k / (mu_q + 1e-6)

    # 时间语境对齐特征（二值化）
    is_same_month = 1.0 if d.get("month") == q.get("month") else 0.0
    is_same_time_of_day = 1.0 if d.get("time_of_day") == q.get("time_of_day") else 0.0
    is_same_weekday = 1.0 if d.get("is_weekend") == q.get("is_weekend") else 0.0

    return np.array([
        distance,           # 0: 向量距离
        abs_mu_diff,       # 1: 均值差异
        abs_sigma_diff,    # 2: 标准差差异
        is_same_month,     # 3: 月份相同
        is_same_time_of_day,# 4: 时段相同
        is_same_weekday,   # 5: 工作日/周末相同
        mu_ratio,          # 6: 均值比率
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 标签构造
# ---------------------------------------------------------------------------

def compute_quality_score(
    predicted_future: list,
    true_future: np.ndarray,
) -> float:
    """
    用 Pearson 相关系数衡量召回片段的未来走向与真实未来走向的相似度。

    数学定义：
        r = cov(predicted_future, true_future) / (std_pred * std_true)

    分数范围：[-1, 1]
        r → 1   : 高度正相关，片段非常有参考价值
        r → 0   : 无相关，片段参考价值低
        r → -1  : 高度负相关，片段有误导性（给予负分）

    若任一序列方差为 0（全相等），返回 0.0。

    参数:
        predicted_future: list[float]，召回片段的 future_y（已反归一化到原始量级）
        true_future:      np.ndarray，真实的未来序列（已反归一化）

    返回:
        float，Pearson 相关系数
    """
    pred_arr = np.asarray(predicted_future, dtype=np.float64)
    true_arr = np.asarray(true_future, dtype=np.float64)

    if len(pred_arr) == 0 or len(true_arr) == 0:
        return 0.0

    # 确保等长
    min_len = min(len(pred_arr), len(true_arr))
    pred_arr = pred_arr[:min_len]
    true_arr = true_arr[:min_len]

    # 方差保护
    std_pred = np.std(pred_arr, ddof=0)
    std_true = np.std(true_arr, ddof=0)
    if std_pred < 1e-10 or std_true < 1e-10:
        return 0.0

    r, _ = pearsonr(pred_arr, true_arr)
    if np.isnan(r):
        return 0.0
    return float(r)


# ---------------------------------------------------------------------------
# 核心训练逻辑
# ---------------------------------------------------------------------------

def sample_training_data(
    df: pd.DataFrame,
    column: str,
    history_len: int,
    future_len: int,
    num_samples: int,
    rng: np.random.Generator,
) -> list:
    """
    从 DataFrame 中随机采样 num_samples 个时间点，
    每个采样点返回 (query_metadata, doc_payload, distance, quality_score) 列表。

    参数:
        df:         ETT DataFrame（包含 date 列和数值列）
        column:     要采样的数值列名
        history_len: 历史窗口长度
        future_len: 未来窗口长度
        num_samples: 采样数量
        rng:        numpy 随机数生成器（用于可复现性）

    返回:
        List of (query_metadata, doc_payload, distance, quality_score) tuples
    """
    total_len = history_len + future_len
    max_start_idx = len(df) - total_len

    if max_start_idx <= 0:
        print(f"  错误: DataFrame 长度不足以生成样本（需要 {total_len}，实际 {len(df)}）")
        return []

    # 随机采样起始位置
    sample_indices = rng.choice(
        max_start_idx,
        size=min(num_samples, max_start_idx),
        replace=False,
    )
    sample_indices = sorted(sample_indices.tolist())

    rows = []
    for idx in sample_indices:
        ts = pd.Timestamp(df["date"].iloc[idx + history_len])
        future_start = idx + history_len

        history_vals = df[column].iloc[idx:future_start].values.astype(np.float64)
        true_future_vals = df[column].iloc[future_start:future_start + future_len].values

        # 处理 NaN
        if np.isnan(history_vals).any() or np.isnan(true_future_vals).any():
            continue

        rows.append({
            "history_vals": history_vals,
            "true_future_vals": true_future_vals,
            "timestamp": ts,
            "start_idx": idx,
        })

    return rows


def build_dataset(
    sample_rows: list,
    processor: TSProcessor,
    encoder: ONNXEncoder,
    retriever: QdrantRetriever,
    top_k: int,
    collection_name: str,
) -> tuple:
    """
    对采样的历史片段进行 Qdrant 检索，构造精排训练数据集。

    对每个采样点：
        1. 归一化 history_vals → 得到 mu_q, sigma_q
        2. ONNX 编码 → query_vector
        3. Qdrant 检索 Top-K → 得到 K 个召回片段
        4. 对每个召回片段，构造交叉特征 + Pearson 质量分数作为标签
        5. 汇总为 (X_features, Y_targets) 数据集

    参数:
        sample_rows:      sample_training_data() 返回的采样行列表
        processor:        TSProcessor 实例
        encoder:          ONNXEncoder 实例
        retriever:        QdrantRetriever 实例
        top_k:            检索的 Top-K
        collection_name:  Qdrant collection 名称

    返回:
        (X: np.ndarray, y: np.ndarray, meta: list)
        X.shape = (N, 7)，y.shape = (N,)
    """
    X_list: list = []
    y_list: list = []
    meta_list: list = []

    total = len(sample_rows)
    for i, row in enumerate(sample_rows):
        history_vals = row["history_vals"]
        true_future = row["true_future_vals"]
        ts = row["timestamp"]

        # 归一化
        normalized, mu_q, sigma_q = processor.normalize(history_vals.tolist())

        # 向量编码
        query_vector = encoder.encode(normalized)

        # 检索
        search_results = retriever.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            query_filter=None,  # 训练时不加 Filter，获取最广泛的候选集
        )

        if not search_results:
            continue

        # Query 元数据
        q_meta = {
            "mu": mu_q,
            "sigma": sigma_q,
            **extract_time_features(ts),
        }

        # 对每个召回片段构造样本
        for result in search_results:
            payload = result.get("payload", {})
            score = result.get("score", 0.0)
            distance = 1.0 - score if score <= 1.0 else 0.0

            # 反归一化 future_y_k → 原始量级
            mu_k = payload.get("mu", 0.0)
            sigma_k = payload.get("sigma", 1.0)
            raw_future = payload.get("future_y", [])
            if sigma_k < 1e-10:
                sigma_k = 1.0
            future_k = [v * sigma_k + mu_k for v in raw_future]

            # 构造特征
            features = _build_cross_features(q_meta, payload, distance)

            # 构造标签（Pearson 相关系数）
            quality_score = compute_quality_score(future_k, true_future)

            X_list.append(features)
            y_list.append(quality_score)
            meta_list.append({
                "sample_idx": i,
                "doc_id": result.get("id"),
                "distance": distance,
                "quality_score": quality_score,
            })

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"    进度: {i+1}/{total} 个采样点已处理")

    if not X_list:
        return np.array([], dtype=np.float32).reshape(0, 7), np.array([], dtype=np.float32), []

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta_list


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> dict:
    """
    训练 XGBoost 精排模型并导出为 JSON。

    参数:
        X:           特征矩阵，shape = (N, 7)
        y:           标签向量，shape = (N,)，Pearson 相关系数 ∈ [-1, 1]
        output_path: 模型输出路径

    返回:
        dict，训练摘要统计
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # 数据划分（80% 训练 / 20% 验证）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n  数据集划分: 训练集 {len(X_train)} 条，验证集 {len(X_val)} 条")

    # 构造 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost 参数（工业级精排配置）
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
        "verbosity": 1,
    }

    evals = [(dtrain, "train"), (dval, "val")]

    print(f"\n  开始训练 XGBoost（max_depth={params['max_depth']}, eta={params['eta']}）...")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    # 在验证集上评估
    y_pred_val = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)

    print(f"\n  验证集 RMSE: {rmse:.4f}")
    print(f"  验证集 R^2:  {r2:.4f}")

    # 特征重要性
    importance = model.get_score(importance_type="gain")
    feature_names = [
        "distance", "abs_mu_diff", "abs_sigma_diff",
        "is_same_month", "is_same_time_of_day",
        "is_same_weekday", "mu_ratio",
    ]
    print("\n  特征重要性（Gain）:")
    for fname in feature_names:
        score = importance.get(fname, 0.0)
        print(f"    {fname:25s}: {score:.4f}")

    # 导出模型（JSON 格式，XGBoost 原生格式）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    print(f"\n  模型已保存: {output_path}")

    # 同时导出一个 meta.json 记录配置信息
    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "num_boost_round": model.best_iteration + 1,
        "best_score": float(model.best_score) if hasattr(model, "best_score") else None,
        "val_rmse": float(rmse),
        "val_r2": float(r2),
        "num_train_samples": int(len(X_train)),
        "num_val_samples": int(len(X_val)),
        "total_samples": int(len(X)),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  训练元信息已保存: {meta_path}")

    return meta


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost 精排模型离线训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        default="ETTh1",
        help="ETT 数据集文件名（不含路径，如 ETTh1）",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="OT",
        help="要采样的数值列名（如 OT, HUFL）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        dest="num_samples",
        help="采样数量（默认 5000）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help="Qdrant 检索的 Top-K（默认 20）",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=cfg.system.input_length,
        dest="history_len",
        help=f"历史窗口长度（默认 {cfg.system.input_length}）",
    )
    parser.add_argument(
        "--future-len",
        type=int,
        default=cfg.system.default_future_length,
        dest="future_len",
        help=f"未来窗口长度（默认 {cfg.system.default_future_length}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新训练（覆盖已存在的模型文件）",
    )
    args = parser.parse_args()

    system_cfg = cfg.system
    ranker_path = system_cfg.get_ranker_path()

    # 检查模型文件
    if ranker_path.exists() and not args.force:
        print(f"\n模型文件已存在: {ranker_path}")
        print("使用 --force 参数可强制重新训练。")
        print("退出。")
        sys.exit(0)

    # Step 1: 加载数据
    ett_file = system_cfg.ett_data_dir / f"{args.file}.csv"
    if not ett_file.exists():
        print(f"\n错误: ETT 数据文件不存在: {ett_file}")
        print(f"请确认 ETT_data/ 目录下有 {args.file}.csv 文件。")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"XGBoost 精排模型训练")
    print(f"{'='*60}")
    print(f"数据集: {args.file}.csv")
    print(f"列:     {args.column}")
    print(f"采样数: {args.num_samples}")
    print(f"Top-K:  {args.top_k}")
    print(f"历史窗口: {args.history_len}，未来窗口: {args.future_len}")
    print(f"{'='*60}")

    print(f"\n[Step 1] 加载 ETT 数据...")
    df = pd.read_csv(ett_file, parse_dates=["date"])
    numeric_cols = [c for c in df.columns if c != "date"]
    if args.column not in numeric_cols:
        print(f"错误: 列 '{args.column}' 不存在，可用列: {numeric_cols}")
        sys.exit(1)
    print(f"  数据形状: {df.shape}")
    print(f"  时间范围: {df['date'].min()} ~ {df['date'].max()}")

    # Step 2: 初始化组件
    print(f"\n[Step 2] 初始化组件...")
    processor = TSProcessor()

    encoder_path = system_cfg.get_encoder_path()
    if not encoder_path.exists():
        print(f"\n  警告: ONNX 模型不存在: {encoder_path}")
        print(f"  将使用 fallback encoder (encoder_v1.onnx)")
        encoder_path = system_cfg.root_dir / "encoder_v1.onnx"

    encoder = ONNXEncoder(onnx_path=str(encoder_path), input_length=args.history_len)
    print(f"  ONNX编码器: {encoder_path.name}，嵌入维度: {encoder.get_embedding_dim()}")

    retriever = QdrantRetriever(storage_path=system_cfg.qdrant_storage_path)
    print(f"  Qdrant存储: {system_cfg.qdrant_storage_path}")

    collection_name = system_cfg.qdrant_collection
    point_count = retriever.count_points(collection_name)
    print(f"  Collection: {collection_name}，向量数: {point_count}")
    if point_count == 0:
        print(f"\n  错误: Qdrant collection 为空，请先运行摄入脚本。")
        print(f"  示例: python scripts/ingest_ett.py --file {args.file} --column {args.column}")
        sys.exit(1)

    # Step 3: 采样
    print(f"\n[Step 3] 采样 {args.num_samples} 个训练样本...")
    rng = np.random.default_rng(seed=42)
    sample_rows = sample_training_data(
        df=df,
        column=args.column,
        history_len=args.history_len,
        future_len=args.future_len,
        num_samples=args.num_samples,
        rng=rng,
    )
    print(f"  有效采样数: {len(sample_rows)}")

    if len(sample_rows) == 0:
        print("  错误: 采样数量为 0，请检查数据是否充足。")
        sys.exit(1)

    # Step 4: 构造训练数据集
    print(f"\n[Step 4] Qdrant 检索 + 构造训练数据集...")
    start_time = time.time()
    X, y, meta = build_dataset(
        sample_rows=sample_rows,
        processor=processor,
        encoder=encoder,
        retriever=retriever,
        top_k=args.top_k,
        collection_name=collection_name,
    )
    elapsed = time.time() - start_time
    print(f"\n  数据集构造完成，耗时 {elapsed:.1f}s")
    print(f"  总样本数: {len(X)}（{len(sample_rows)} 个采样点 × {args.top_k} 个召回）")

    if len(X) == 0:
        print("  错误: 训练数据集为空（所有采样点均未召回任何结果）。")
        sys.exit(1)

    # 打印标签分布
    print(f"\n  标签（Pearson 相关系数）分布:")
    print(f"    min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")
    positive_ratio = float((y > 0).sum() / len(y))
    print(f"    正相关比例: {positive_ratio:.1%}")

    # Step 5: 训练
    print(f"\n[Step 5] 训练 XGBoost 精排模型...")
    meta_info = train_xgboost(X, y, ranker_path)

    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"{'='*60}")
    print(f"模型路径: {ranker_path}")
    print(f"特征维度: 7 维（distance, abs_mu_diff, abs_sigma_diff,")
    print(f"           is_same_month, is_same_time_of_day,")
    print(f"           is_same_weekday, mu_ratio）")
    print(f"验证集 RMSE: {meta_info.get('val_rmse', 'N/A'):.4f}")
    print(f"验证集 R^2:  {meta_info.get('val_r2', 'N/A'):.4f}")
    print(f"\n【下一步】")
    print(f"  1. 重启 FastAPI 服务（自动加载新模型）")
    print(f"  2. 调用 POST /api/v2/agent/forecast 验证精排效果")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
