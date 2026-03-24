from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional

from app.schemas.event import EventSearchRequest, EventSearchResponse, EventItem
from app.services.event_service import EventService
from app.clients.tavily_client import TavilyClient

# 创建路由实例
router = APIRouter(
    prefix="/api/events",
    tags=["events"],
    responses={404: {"description": "Not found"}},
)

# -----------------------------------------------------------------------------
# 依赖注入 (Dependency Injection Helper)
# -----------------------------------------------------------------------------
# 实际生产中可能包含数据库 Session 或 Config
def get_tavily_client() -> TavilyClient:
    """提供 TavilyClient 实例"""
    return TavilyClient()

def get_event_service(tavily_client: TavilyClient = Depends(get_tavily_client)) -> EventService:
    """提供 EventService 实例，注入 client"""
    return EventService(tavily_client)

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@router.get("/search", response_model=EventSearchResponse)
async def search_events(
    # 使用 Depends 将 query parameters 映射到 Pydantic 模型
    # FastAPI 会自动解析 request parameters 到 EventSearchRequest 的字段
    search_params: EventSearchRequest = Depends(),
    service: EventService = Depends(get_event_service)
) -> EventSearchResponse:
    """
    根据关键词和日期范围搜索新闻事件 (Search API)
    
    Args:
        search_params (EventSearchRequest): 包含 query, start_date, end_date
        service (EventService): 业务服务层注入

    Returns:
        EventSearchResponse: 包含搜索结果列表
    """
    try:
        # 调用服务层处理业务逻辑
        # service 层会调用 client 获取数据并进行转换
        response = await service.fetch_news_events(search_params)
        return response
    except ValueError as e:
        # 捕获已知业务异常
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 捕获未知异常
        # 可以在这里记录日志 (logger.error(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
