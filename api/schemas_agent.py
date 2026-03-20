"""
Pydantic 数据模型 — Layer 2 Agentic API (api/routes_agent.py)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class AgentForecastRequest(BaseModel):
    """
    Layer 2 Agentic Forecast 请求

    用户以自然语言描述查询场景（如"夏季周末用电高峰"），
    系统通过以下链路响应：
        1. LLM 解析意图 → Qdrant Filter 条件
        2. 结合语义向量检索 + 时间特征过滤
        3. IDW/XGBoost 融合预测
        4. LLM 生成结构化分析报告
    """

    user_query: str = Field(
        ...,
        description="用户的自然语言查询",
        min_length=2,
        max_length=500,
        examples=[
            "结合当前夏季周末的用电高峰，预测未来趋势",
            "周末晚上高油温会怎么走",
            "工作日早晨用电低谷预测",
        ],
    )
    history_x: List[float] = Field(
        ...,
        description="历史时序数据，长度应与模型期望长度（L=100）匹配",
        min_length=50,
        max_length=500,
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="向量检索返回的最近邻数量（建议 5~20）",
    )
    include_raw_prediction: bool = Field(
        default=False,
        description="是否在响应中包含原始数值预测（不含 LLM 报告）",
    )


# ---------------------------------------------------------------------------
# 内部链路结果模型（用于响应嵌套）
# ---------------------------------------------------------------------------

class TimeFeatureSummary(BaseModel):
    """召回片段的时间特征摘要"""

    month: Optional[int] = Field(default=None, description="1-12")
    hour: Optional[int] = Field(default=None, description="0-23")
    is_weekend: Optional[bool] = Field(default=None)
    time_of_day: Optional[str] = Field(
        default=None,
        description="night | morning | afternoon | evening",
    )


class RetrievedChunk(BaseModel):
    """召回的单个历史片段摘要"""

    id: int = Field(..., description="Qdrant 中的向量 ID")
    score: float = Field(..., description="向量相似度分数 (0~1)")
    future_y: List[float] = Field(
        default_factory=list,
        description="该片段对应的未来真实序列（已反归一化）",
    )
    mu: Optional[float] = Field(default=None, description="Z-Score 均值")
    sigma: Optional[float] = Field(default=None, description="Z-Score 标准差")
    time_features: TimeFeatureSummary = Field(
        default_factory=TimeFeatureSummary,
        description="时间特征",
    )
    source: Optional[str] = Field(default=None, description="数据列名")


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------

class AgentForecastResponse(BaseModel):
    """
    Layer 2 Agentic Forecast 响应

    包含:
        - intent_filter:    LLM 解析出的 Qdrant Filter 条件（透明展示）
        - prediction:        IDW/XGBoost 融合后的预测序列（已反归一化）
        - retrieved_chunks: Top-K 召回片段的元数据摘要
        - report:           LLM 生成的深度分析报告（Markdown 格式）
    """

    success: bool = Field(..., description="请求是否成功")
    intent_filter: dict = Field(
        ...,
        description="LLM 解析出的 Qdrant Filter 条件，可用于 Debug",
    )
    prediction: List[float] = Field(
        default_factory=list,
        description="融合预测序列（已反归一化）",
    )
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Top-K 召回片段的元数据摘要",
    )
    report: str = Field(
        default="",
        description="LLM 生成的深度分析报告（Markdown 格式）",
    )
    message: str = Field(
        default="",
        description="补充说明或错误信息",
    )


class IntentParseRequest(BaseModel):
    """独立意图解析接口请求（仅调用 LLM，不走向量检索链路）"""

    user_query: str = Field(
        ...,
        description="用户的自然语言查询",
        min_length=2,
        max_length=500,
    )


class IntentParseResponse(BaseModel):
    """独立意图解析接口响应"""

    success: bool = Field(..., description="解析是否成功")
    intent_filter: dict = Field(
        ...,
        description="Qdrant Filter JSON",
    )
    message: str = Field(default="", description="补充说明")
