from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import date

# ==============================================================================
# 请求模型定义
# 用于接收前端传递的查询参数
# ==============================================================================
class EventSearchRequest(BaseModel):
    """
    事件搜索请求参数模型
    由于使用的是 GET 请求，这些字段将通过 Query Parameters 传递
    """
    query: str = Field(..., description="搜索关键词，必填")
    #Start date and end date are strings because query params come as strings usually, 
    #but Pydantic can parse dates automatically if format is ISO8601 (YYYY-MM-DD).
    start_date: Optional[date] = Field(None, description="搜索开始日期 (格式: YYYY-MM-DD)，选填")
    end_date: Optional[date] = Field(None, description="搜索结束日期 (格式: YYYY-MM-DD)，选填")

# ==============================================================================
# 响应模型定义
# 用于规范返回给前端的数据结构
# ==============================================================================
class EventItem(BaseModel):
    """单个事件/新闻条目模型"""
    title: str = Field(..., description="新闻标题")
    url: str = Field(..., description="原文链接")
    content: str = Field(..., description="新闻摘要或内容片段")
    published_date: Optional[str] = Field(None, description="发布时间")
    # source: Optional[str] = Field(None, description="来源") # 可选扩展

class EventSearchResponse(BaseModel):
    """搜索结果响应模型"""
    total_results: int = Field(0, description="搜索到的结果总数")
    results: List[EventItem] = Field(default=[], description="新闻列表")


class EventAlignmentRequest(BaseModel):
    """事件与 Polymarket 波动对齐请求"""
    query: str = Field(..., description="事件关键词，必填")
    start_date: Optional[date] = Field(None, description="开始日期 (YYYY-MM-DD)")
    end_date: Optional[date] = Field(None, description="结束日期 (YYYY-MM-DD)")
    news_limit: int = Field(10, ge=1, le=50, description="新闻结果上限")
    event_limit: int = Field(20, ge=1, le=100, description="Polymarket 事件候选上限")
    fidelity: int = Field(60, ge=1, le=1440, description="价格点间隔(秒)")


class PolymarketEventSummary(BaseModel):
    id: str
    title: str
    slug: Optional[str] = None
    category: Optional[str] = None
    market_id: Optional[str] = None
    market_question: Optional[str] = None
    token_id: Optional[str] = None


class MarketTimePoint(BaseModel):
    timestamp: int = Field(..., description="Unix 时间戳(秒)")
    datetime: str = Field(..., description="ISO8601 UTC 时间")
    price: float = Field(..., description="Yes 概率，范围 0~1")


class AlignedEventPoint(BaseModel):
    title: str
    url: str
    news_time: Optional[str] = None
    market_timestamp: Optional[int] = None
    market_datetime: Optional[str] = None
    market_price: Optional[float] = None


class EventAlignmentResponse(BaseModel):
    query: str
    selected_event: Optional[PolymarketEventSummary] = None
    market_series: List[MarketTimePoint] = Field(default=[], description="市场时间序列")
    news: List[EventItem] = Field(default=[], description="相关新闻")
    aligned_events: List[AlignedEventPoint] = Field(default=[], description="对齐结果")
    note: Optional[str] = None
