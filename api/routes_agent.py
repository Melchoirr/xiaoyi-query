"""
Layer 2 Agentic Router — FastAPI 路由层

Endpoints:
    POST /api/v2/agent/forecast
        完整 Agentic 链路：意图解析 → 过滤检索 → IDW融合 → 报告生成

    POST /api/v2/agent/intent
        独立意图解析接口：仅调用 LLM 将自然语言映射为 Qdrant Filter

依赖注入说明:
    Layer-1 单例通过 inject_layer1_instances() 注入（由 main.py lifespan 调用）。
    TimeRAGAgent 通过 get_agent() 延迟构造，OPENAI_API_KEY 必须已配置。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas_agent import (
    AgentForecastRequest,
    AgentForecastResponse,
    RetrievedChunk,
    IntentParseRequest,
    IntentParseResponse,
    TimeFeatureSummary,
)
from components.agent import (
    TimeRAGAgent,
    IntentParseError,
    ReportGenerationError,
)
from core.processor import TSProcessor
from components.encoder import ONNXEncoder
from components.retriever import QdrantRetriever
from components.ranker import FusionRanker
from core.config import cfg


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/agent", tags=["Layer 2 — Agentic"])

# 从 config.py 读取（与 main.py 保持一致）
_input_length = cfg.system.input_length
_collection_name = cfg.system.qdrant_collection


# ---------------------------------------------------------------------------
# Layer-1 单例注入（由 main.py lifespan 调用）
# ---------------------------------------------------------------------------
_processor_instance: TSProcessor | None = None
_encoder_instance: ONNXEncoder | None = None
_retriever_instance: QdrantRetriever | None = None
_ranker_instance: FusionRanker | None = None


def inject_layer1_instances(
    processor: TSProcessor,
    encoder: ONNXEncoder,
    retriever: QdrantRetriever,
    ranker: FusionRanker,
) -> None:
    """
    供 main.py lifespan 调用，将已初始化的 Layer-1 单例注入本模块。
    避免每个请求重复创建 ONNX Session / Qdrant Client。
    """
    global _processor_instance, _encoder_instance, _retriever_instance, _ranker_instance
    _processor_instance = processor
    _encoder_instance = encoder
    _retriever_instance = retriever
    _ranker_instance = ranker


def get_processor() -> TSProcessor:
    if _processor_instance is None:
        raise RuntimeError(
            "Layer-1 单例未注入。请确保 main.py lifespan 已执行 inject_layer1_instances。"
        )
    return _processor_instance


def get_encoder() -> ONNXEncoder:
    if _encoder_instance is None:
        raise RuntimeError(
            "Layer-1 单例未注入。请确保 main.py lifespan 已执行 inject_layer1_instances。"
        )
    return _encoder_instance


def get_retriever() -> QdrantRetriever:
    if _retriever_instance is None:
        raise RuntimeError(
            "Layer-1 单例未注入。请确保 main.py lifespan 已执行 inject_layer1_instances。"
        )
    return _retriever_instance


def get_ranker() -> FusionRanker:
    if _ranker_instance is None:
        raise RuntimeError(
            "Layer-1 单例未注入。请确保 main.py lifespan 已执行 inject_layer1_instances。"
        )
    return _ranker_instance


# ---------------------------------------------------------------------------
# Agent 实例（延迟构造，避免 startup 时无 API Key 直接崩溃）
# ---------------------------------------------------------------------------
_agent_instance: TimeRAGAgent | None = None


def get_agent() -> TimeRAGAgent:
    """获取 TimeRAGAgent 实例（延迟构造）。"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TimeRAGAgent()
    return _agent_instance


def set_agent(agent: TimeRAGAgent) -> None:
    """供测试注入 mock agent。"""
    global _agent_instance
    _agent_instance = agent


# ---------------------------------------------------------------------------
# Endpoint: POST /api/v2/agent/forecast
# ---------------------------------------------------------------------------

@router.post(
    "/forecast",
    response_model=AgentForecastResponse,
    summary="Agentic 预测（自然语言 → 过滤检索 → 数值预测 → 报告）",
    description="""
完整 Agentic 预测链路：

1. **意图解析** — 调用 LLM 将 `user_query` 映射为 Qdrant Filter 条件
2. **向量检索** — 将 `history_x` 归一化 → ONNX 编码 → Qdrant 向量检索
3. **精排融合（Layer 3）** — XGBoost 精排模型打分 + Softmax 融合
4. **报告生成** — 调用 LLM 结合预测数值和召回元数据生成结构化报告

**示例请求:**

```json
{
    "user_query": "结合当前夏季周末的用电高峰，预测未来趋势",
    "history_x": [30.1, 30.5, ...],
    "top_k": 10
}
```

**前置条件:**
- LLM API Key 已配置（支持 OpenAI / DeepSeek / Qwen / Ollama，见 core/config.py）
- Qdrant 中已摄入包含时间特征的向量数据
- 可选：models/xgb_ranker.json 已训练（未训练时自动降级为 IDW 融合）
    """,
    responses={
        200: {"description": "预测成功"},
        400: {"description": "输入校验失败"},
        500: {"description": "LLM 调用或内部错误"},
    },
)
async def agent_forecast(
    processor: TSProcessor = Depends(get_processor),
    encoder: ONNXEncoder = Depends(get_encoder),
    retriever: QdrantRetriever = Depends(get_retriever),
    ranker: FusionRanker = Depends(get_ranker),
    agent: TimeRAGAgent = Depends(get_agent),
    request: AgentForecastRequest = None,
) -> AgentForecastResponse:
    """
    Agentic 时序预测接口 — 完整编排链路（Step 1~8）：

    Step 1: TimeRAGAgent.parse_intent(user_query)
        调用 LLM 将自然语言查询解析为 Qdrant Filter 条件。
        例: "周末晚上用电高峰" → {"must": [{"key":"is_weekend","match":{"value":true}},
                                            {"key":"time_of_day","match":{"value":"evening"}}]}

    Step 2: TSProcessor.normalize(history_x)
        对输入的 100 维历史序列做 Z-Score 归一化，返回 (normalized, mu, sigma)。

    Step 3: ONNXEncoder.encode(normalized)
        调用预训练 Foundation Model ONNX 模型，将归一化序列编码为 256 维稠密向量。

    Step 4: QdrantRetriever.search(query_vector, query_filter)
        以向量相似度为主、Filter 条件为辅，执行混合检索，召回 Top-K 相关片段。
        若 Filter 过严导致零召回，自动降级为无过滤检索（保证有结果可用）。

    Step 5: FusionRanker.rank_and_fuse(query_metadata, search_results, future_length)
        Layer 3 精排融合：XGBoost 精排模型对 Top-K 片段打质量分，Softmax 归一化为概率权重，
        加权求和输出融合预测 y_hat。反归一化: y_hat = fused_normalized * sigma + mu。

    Step 6: Payload 反归一化
        将每条召回记录的 future_y（存储在归一化空间）用自己的 mu/sigma 还原，
        用于 Step 7 的报告生成。

    Step 7: TimeRAGAgent.generate_report(user_query, fused_prediction, retrieved_metadata)
        调用 LLM，结合原始问题、预测数值和召回片段元数据，
        生成具有解释性的专业分析报告（Markdown 格式）。

    Step 8: 返回统一 JSON 响应
        包含 prediction_values（数值预测）和 ai_analysis_report（文字解析）两个字段。

    参数:
        request: AgentForecastRequest
        processor: TSProcessor — Z-Score 归一化 / 反归一化
        encoder: ONNXEncoder — 预训练 Foundation Model 向量化
        retriever: QdrantRetriever — 向量 + Payload 过滤检索
        ranker: FusionRanker — Layer 3 XGBoost 精排 + Softmax 融合（未训练时降级为 IDW）
        agent: TimeRAGAgent — LLM 意图解析 + 报告生成
    """
    # ------------------------------------------------------------------
    # Step 1: 意图解析 — 自然语言 → Qdrant Filter
    # ------------------------------------------------------------------
    try:
        intent_filter = agent.parse_intent(request.user_query)
    except IntentParseError as exc:
        logger.error(f"意图解析失败: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM 意图解析失败: {exc}",
        )

    # ------------------------------------------------------------------
    # Step 2: 输入校验 + 归一化
    # ------------------------------------------------------------------
    history = request.history_x
    if len(history) != _input_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"history_x 长度必须为 {_input_length}，实际 {len(history)}",
        )

    normalized, mu, sigma = processor.normalize(history)

    # ------------------------------------------------------------------
    # Step 3: 向量编码
    # ------------------------------------------------------------------
    query_vector = encoder.encode(normalized)

    # ------------------------------------------------------------------
    # Step 4: Qdrant 检索（使用 LLM 解析出的 Filter）
    # ------------------------------------------------------------------
    has_filter = (
        intent_filter.get("must")
        or intent_filter.get("should")
        or intent_filter.get("must_not")
    )

    search_results = retriever.search(
        collection_name=_collection_name,
        query_vector=query_vector,
        top_k=request.top_k,
        query_filter=intent_filter if has_filter else None,
    )

    # ------------------------------------------------------------------
    # Step 5: IDW 融合预测
    # ------------------------------------------------------------------
    if not search_results:
        # Filter 过严导致无召回，降级为无过滤检索（兜底）
        logger.warning(
            f"Filter 导致无召回，降级为无过滤检索。"
            f"filter={intent_filter}"
        )
        search_results = retriever.search(
            collection_name=_collection_name,
            query_vector=query_vector,
            top_k=request.top_k,
            query_filter=None,
        )

    # 确定预测长度（取第一条召回的 future_y 长度）
    future_length = 48
    if search_results:
        first_payload = search_results[0].get("payload", {})
        first_future = first_payload.get("future_y", [])
        if first_future:
            future_length = len(first_future)

    # Step 5: Layer 3 精排融合预测（XGBoost 精排 + Softmax）
    # 构造 query_metadata，传入 rank_and_fuse 用于交叉特征构造
    query_metadata = {
        "mu": mu,
        "sigma": sigma,
    }

    fused_normalized = ranker.rank_and_fuse(
        query_metadata=query_metadata,
        retrieved_items=search_results,
        future_length=future_length,
    )

    # 反归一化（将预测从归一化空间还原到原始量级）
    denormed_prediction = processor.denormalize(fused_normalized, mu, sigma)
    prediction_list: List[float] = (
        denormed_prediction.tolist()
        if hasattr(denormed_prediction, "tolist")
        else list(denormed_prediction)
    )

    # ------------------------------------------------------------------
    # Step 6: 召回片段 Payload 反归一化
    #    future_y 在 Qdrant 中存的是归一化值，
    #    需要用每条记录自己的 mu/sigma 还原为原始量级再传给 LLM
    # ------------------------------------------------------------------
    def _denorm(values: List[float], m: float, s: float) -> List[float]:
        if s < 1e-10:
            return values
        return [round(v * s + m, 4) for v in values]

    retrieved_chunks: List[RetrievedChunk] = []
    for item in search_results:
        payload = item.get("payload", {})
        m = payload.get("mu", 0.0)
        s = payload.get("sigma", 1.0)
        raw_future = _safe_list(payload.get("future_y", []))
        denormed_future = _denorm(raw_future, m, s)

        retrieved_chunks.append(
            RetrievedChunk(
                id=item.get("id", 0),
                score=round(float(item.get("score", 0.0)), 4),
                future_y=denormed_future,
                mu=payload.get("mu"),
                sigma=payload.get("sigma"),
                time_features=TimeFeatureSummary(
                    month=payload.get("month"),
                    hour=payload.get("hour"),
                    is_weekend=payload.get("is_weekend"),
                    time_of_day=payload.get("time_of_day"),
                ),
                source=payload.get("source"),
            )
        )

    # ------------------------------------------------------------------
    # Step 7: 生成 LLM 报告
    # ------------------------------------------------------------------
    ai_report = ""
    try:
        ai_report = agent.generate_report(
            user_query=request.user_query,
            fused_prediction=prediction_list,
            retrieved_metadata=[
                {
                    "score": chunk.score,
                    "payload": {
                        "future_y": chunk.future_y,
                        "month": chunk.time_features.month,
                        "hour": chunk.time_features.hour,
                        "is_weekend": chunk.time_features.is_weekend,
                        "time_of_day": chunk.time_features.time_of_day,
                    },
                }
                for chunk in retrieved_chunks
            ],
            temperature=0.3,
        )
    except ReportGenerationError as exc:
        logger.warning(f"报告生成失败，降级为空报告: {exc}")
        ai_report = (
            "[报告生成失败] LLM 调用失败，请检查 API Key 或稍后重试。"
            f"\n原始错误: {exc}"
        )
    except Exception as exc:
        logger.error(f"报告生成未知错误: {exc}")
        ai_report = f"[报告生成失败] 未知错误: {exc}"

    # ------------------------------------------------------------------
    # Step 8: 构建统一响应（prediction_values + ai_analysis_report 均始终返回）
    # ------------------------------------------------------------------
    return AgentForecastResponse(
        success=True,
        intent_filter=intent_filter,
        prediction_values=prediction_list,
        retrieved_chunks=retrieved_chunks,
        ai_analysis_report=ai_report,
        message=f"基于 Top-{len(search_results)} 相似片段{'（含过滤）' if has_filter else '（无过滤）'}预测，共召回 {len(search_results)} 条",
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /api/v2/agent/intent
# ---------------------------------------------------------------------------

@router.post(
    "/intent",
    response_model=IntentParseResponse,
    summary="独立意图解析（仅调用 LLM）",
    description="""
独立意图解析接口，仅将自然语言映射为 Qdrant Filter JSON，不涉及向量检索。
可用于调试 Prompt 效果，或在调用方自行处理 Filter 逻辑时使用。

**示例请求:**

```json
{
    "user_query": "周末晚上高油温会怎么走"
}
```

**示例响应:**

```json
{
    "success": true,
    "intent_filter": {
        "must": [
            {"key": "is_weekend",  "match": {"value": true}},
            {"key": "time_of_day", "match": {"value": "evening"}}
        ],
        "should": [],
        "must_not": []
    },
    "message": ""
}
```
    """,
    responses={
        200: {"description": "解析成功"},
        500: {"description": "LLM 调用失败"},
    },
)
async def parse_intent(
    request: IntentParseRequest,
    agent: TimeRAGAgent = Depends(get_agent),
) -> IntentParseResponse:
    """
    独立意图解析接口。

    仅调用 TimeRAGAgent.parse_intent()，返回 Qdrant Filter JSON。
    不走向量检索链路，适合调试和独立使用。
    """
    try:
        intent_filter = agent.parse_intent(request.user_query)
        return IntentParseResponse(
            success=True,
            intent_filter=intent_filter,
            message="解析成功",
        )
    except IntentParseError as exc:
        logger.error(f"意图解析失败: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"意图解析失败: {exc}",
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_list(value: Any) -> List[Any]:
    """防御性列表转换（用于反归一化前预处理）。"""
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None:
        return []
    return [value]
