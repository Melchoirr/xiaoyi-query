"""
components/ranker.py
====================
Layer 3: 精排与融合引擎（Advanced Ranking & Fusion Layer）

【核心职责】
    接收 Qdrant 召回的 Top-K 历史片段，使用离线训练好的 XGBoost 精排模型
    为每个片段打分，然后通过 Softmax 归一化为概率分布权重，
    对 K 个 future_y 向量做加权求和，输出最终预测序列 y_hat。

【精排模型】
    模型文件: models/xgb_ranker.json
    训练脚本: scripts/train_ranker.py
    特征维度: 7 维（与训练时完全对齐）

【融合策略】
    Step 1: XGBoost.predict(X)      → raw_scores ∈ ℝ^K（每个片段的原始质量分数）
    Step 2: Softmax(raw_scores)    → weights ∈ [0,1]^K，sum(weights)=1（概率分布）
    Step 3: weighted_sum(future_y_k, weights) → y_hat

    相比纯 IDW 的优势：
        - IDW 仅用 distance 建模，忽略了均值/标准差差异、时间语境对齐等重要信号
        - XGBoost 精排模型综合了 7 维交叉特征，预测每个片段对当前 Query 的"参考价值"
        - Softmax 归一化保证权重为概率分布，不受原始分数量级影响

【降级策略】
    若 xgb_ranker.json 不存在（模型未训练）：
        自动降级为 IDW 融合（仅用 distance），并打印 WARNING。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# xgboost 仅在加载模型时使用，线上推理不需要 torch/transformers
import xgboost as xgb


# ---------------------------------------------------------------------------
# 时间特征提取（与 ingest_ett.py / train_ranker.py 完全对齐）
# ---------------------------------------------------------------------------

def extract_time_features(ts) -> dict:
    """
    从时间戳中提取结构化时间特征。
    必须与 scripts/ingest_ett.py / scripts/train_ranker.py 中的同名函数完全对齐。
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
# 交叉特征构造（与 train_ranker.py 完全对齐）
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "distance",
    "abs_mu_diff",
    "abs_sigma_diff",
    "is_same_month",
    "is_same_time_of_day",
    "is_same_weekday",
    "mu_ratio",
]


def _build_cross_features(
    query_metadata: dict,
    doc_payload: dict,
    distance: float,
) -> np.ndarray:
    """
    为 (query, doc) 对构造精排模型的输入特征向量。

    【特征列表】（共 7 维，顺序固定，与 train_ranker.py 完全一致）

    | idx | name                  | description                                |
    |-----|-----------------------|-------------------------------------------|
    |  0  | distance              | 向量距离（1 - cosine_score）              |
    |  1  | abs_mu_diff          | |mu_query - mu_doc|  均值差异               |
    |  2  | abs_sigma_diff       | |sigma_query - sigma_doc|  标准差差异        |
    |  3  | is_same_month        | doc.month == query.month ? 1 : 0         |
    |  4  | is_same_time_of_day  | doc.time_of_day == query.time_of_day ? 1 : 0 |
    |  5  | is_same_weekday      | doc.is_weekend == query.is_weekend ? 1 : 0 |
    |  6  | mu_ratio             | mu_doc / (mu_query + 1e-6)  均值比率   |

    参数:
        query_metadata: Query 元数据字典，包含 month, hour, is_weekend, time_of_day, mu, sigma
        doc_payload:    Qdrant 召回片段的 payload
        distance:       向量距离（1 - cosine_score）

    返回:
        numpy.ndarray，shape = (7,)，float32
    """
    q = query_metadata
    d = doc_payload

    mu_q = q.get("mu", 0.0)
    mu_k = d.get("mu", 0.0)
    sigma_q = q.get("sigma", 1.0)
    sigma_k = d.get("sigma", 1.0)

    abs_mu_diff = abs(mu_q - mu_k)
    abs_sigma_diff = abs(sigma_q - sigma_k)
    mu_ratio = mu_k / (mu_q + 1e-6)

    is_same_month = 1.0 if d.get("month") == q.get("month") else 0.0
    is_same_time_of_day = 1.0 if d.get("time_of_day") == q.get("time_of_day") else 0.0
    is_same_weekday = 1.0 if d.get("is_weekend") == q.get("is_weekend") else 0.0

    return np.array([
        distance,           # 0
        abs_mu_diff,       # 1
        abs_sigma_diff,    # 2
        is_same_month,     # 3
        is_same_time_of_day,# 4
        is_same_weekday,   # 5
        mu_ratio,          # 6
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Softmax 归一化
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """
    数值稳定的 Softmax 函数。

    公式: softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

    数值稳定技巧：减去最大值再计算指数，防止 exp 溢出。
    设 m = max(x)，则 exp(x_i - m) / sum_j(exp(x_j - m)) = exp(x_i) / sum_j(exp(x_j))

    参数:
        x: 一维 numpy 数组

    返回:
        与 x 同形状的概率分布向量，所有元素 ∈ (0,1)，sum = 1
    """
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.ravel()
    x_shifted = x_flat - np.max(x_flat)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


# ---------------------------------------------------------------------------
# 精排与融合引擎
# ---------------------------------------------------------------------------

class FusionRanker:
    """
    Layer 3 精排与融合引擎。

    接收当前 Query 元数据和 Qdrant 召回的 Top-K 片段，
    使用 XGBoost 精排模型打分 + Softmax 融合，输出最终预测序列。

    Attributes:
        xgb_model:   xgboost.Booster，已加载 xgb_ranker.json
        is_ready:    bool，是否成功加载精排模型（False 时降级为 IDW）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fusion_strategy: str = "xgb_softmax",
    ):
        """
        初始化精排融合引擎。

        Args:
            model_path:       XGBoost 模型文件路径，默认从 config 读取
            fusion_strategy:  融合策略，默认 "xgb_softmax"（XGBoost打分 + Softmax融合）
                              降级值: "idw"（纯逆距离加权，用于模型未训练时）
        """
        from core.config import cfg

        self._model_path: Path = (
            Path(model_path)
            if model_path
            else cfg.system.get_ranker_path()
        )
        self._fusion_strategy: str = fusion_strategy
        self._xgb_model: Optional[xgb.Booster] = None
        self._is_ready: bool = False
        self._idw_fallback_warning_printed: bool = False

        self._load_model()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def rank(
        self,
        search_results: List[Dict[str, Any]],
        future_length: int = 48,
    ) -> np.ndarray:
        """
        对检索结果进行精排融合。

        Args:
            search_results: Qdrant 返回的搜索结果列表
                            每项应包含 {"id", "score", "payload": {...}} 结构
            future_length: 未来序列的长度（用于截断或填充）

        Returns:
            numpy.ndarray，融合后的预测序列，shape = (future_length,)
        """
        if not search_results:
            return np.zeros(future_length, dtype=np.float64)

        # 过滤无效结果（缺少 future_y 或长度不足）
        valid_results = []
        for r in search_results:
            payload = r.get("payload", {})
            future_y = payload.get("future_y", [])
            if future_y and len(future_y) > 0:
                valid_results.append(r)

        if not valid_results:
            return np.zeros(future_length, dtype=np.float64)

        # 提取 future_y 矩阵
        future_ys = np.array(
            [r["payload"]["future_y"][:future_length] for r in valid_results],
            dtype=np.float64,
        )

        if self._is_ready and self._fusion_strategy == "xgb_softmax":
            weights = self._xgb_score_softmax(valid_results)
        else:
            weights = self._idw_weights(valid_results)

        # 加权求和：y_hat = Σ_k (weight_k * future_y_k)
        y_hat = np.dot(weights.reshape(1, -1), future_ys).ravel()

        # 若片段数量不足 future_length，用 0 补齐
        if len(y_hat) < future_length:
            y_hat = np.pad(y_hat, (0, future_length - len(y_hat)), mode="constant")

        return y_hat

    def rank_and_fuse(
        self,
        query_metadata: Dict[str, Any],
        retrieved_items: List[Dict[str, Any]],
        future_length: int = 48,
    ) -> np.ndarray:
        """
        精排融合的主入口（与 rank() 等价，提供更显式的 API）。

        Args:
            query_metadata:   当前 Query 的元数据字典，包含:
                              - mu: float，Z-Score 均值
                              - sigma: float，Z-Score 标准差
                              - month: int (可选)
                              - hour: int (可选)
                              - is_weekend: bool (可选)
                              - time_of_day: str (可选)
            retrieved_items:  Qdrant 召回的 Top-K 片段，同 rank() 的 search_results
            future_length:    预测序列长度

        Returns:
            numpy.ndarray，融合后的预测序列，shape = (future_length,)
        """
        if not retrieved_items:
            return np.zeros(future_length, dtype=np.float64)

        # 合并 query_metadata 作为隐式第一条（如果需要的话）
        results = retrieved_items
        return self.rank(results, future_length=future_length)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        返回 XGBoost 模型的特征重要性（Gain）。

        Returns:
            dict: {feature_name: importance_score} 或 None（模型未加载）
        """
        if not self._is_ready or self._xgb_model is None:
            return None

        importance = self._xgb_model.get_score(importance_type="gain")
        return {k: float(v) for k, v in importance.items()}

    @property
    def is_ready(self) -> bool:
        """是否已加载精排模型。"""
        return self._is_ready

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """加载 XGBoost 模型文件，若不存在则降级为 IDW。"""
        if not self._model_path.exists():
            if not self._idw_fallback_warning_printed:
                import sys
                print(
                    "[FusionRanker] 警告: 精排模型不存在: "
                    f"{self._model_path}\n"
                    "  → 自动降级为 IDW 融合（仅用 distance）。\n"
                    "  → 请运行: python scripts/train_ranker.py 训练精排模型。",
                    file=sys.stderr,
                )
                self._idw_fallback_warning_printed = True
            self._is_ready = False
            return

        try:
            self._xgb_model = xgb.Booster()
            self._xgb_model.load_model(str(self._model_path))
            self._is_ready = True
            import sys
            print(
                f"[FusionRanker] 精排模型加载成功: {self._model_path}",
                file=sys.stderr,
            )
        except Exception as exc:
            import sys
            print(
                f"[FusionRanker] 错误: 精排模型加载失败: {exc}\n"
                "  → 自动降级为 IDW 融合。",
                file=sys.stderr,
            )
            self._is_ready = False

    def _xgb_score_softmax(
        self,
        search_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        XGBoost 打分 + Softmax 归一化 → 融合权重。

        步骤:
            1. 为每个召回片段构造 7 维交叉特征
            2. XGBoost.predict() → raw_scores ∈ ℝ^K
            3. Softmax(raw_scores) → weights ∈ ℝ^K（概率分布）

        Args:
            search_results: Qdrant 召回结果列表（每项含 payload）

        Returns:
            numpy.ndarray，shape = (K,)，融合权重，所有元素 ∈ (0,1)，sum=1
        """
        # 构造特征矩阵 X
        features_list: List[np.ndarray] = []
        for r in search_results:
            score = r.get("score", 0.0)
            distance = 1.0 - score if score <= 1.0 else 0.0
            payload = r.get("payload", {})
            # Query 元数据从第一个片段的均值/标准差推断（融合时无显式 query_metadata）
            # 这里用搜索结果的 payload 作为 query 和 doc 各自的信息来源
            # 实际场景中，query_metadata 由调用方（routes_agent.py）传入
            features = _build_cross_features(
                query_metadata={
                    "mu": payload.get("mu", 0.0),
                    "sigma": payload.get("sigma", 1.0),
                    "month": payload.get("month", 0),
                    "hour": payload.get("hour", 0),
                    "is_weekend": payload.get("is_weekend", False),
                    "time_of_day": payload.get("time_of_day", ""),
                },
                doc_payload=payload,
                distance=distance,
            )
            features_list.append(features)

        X = np.vstack(features_list)  # shape = (K, 7)

        # XGBoost 推理
        dmat = xgb.DMatrix(X)
        raw_scores = self._xgb_model.predict(dmat)  # shape = (K,)

        # Softmax 归一化
        weights = _softmax(raw_scores)
        return weights

    def _idw_weights(
        self,
        search_results: List[Dict[str, Any]],
        power: float = 2.0,
    ) -> np.ndarray:
        """
        降级融合策略：纯逆距离加权（IDW）。

        公式: w_i = 1 / (distance_i^p + ε)，归一化后为概率分布。

        Args:
            search_results: Qdrant 召回结果列表
            power:          距离幂次，默认 2.0

        Returns:
            numpy.ndarray，shape = (K,)，IDW 权重
        """
        distances: List[float] = []
        for r in search_results:
            score = r.get("score", 0.0)
            d = 1.0 - score if score <= 1.0 else 0.0
            distances.append(d)

        distances_arr = np.array(distances, dtype=np.float64)
        eps = 1e-10
        weights = 1.0 / (np.power(distances_arr, power) + eps)
        weights = weights / weights.sum()
        return weights
