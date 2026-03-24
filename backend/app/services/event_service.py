from typing import List
from datetime import datetime

from app.clients.tavily_client import TavilyClient
from app.schemas.event import EventItem, EventSearchRequest, EventSearchResponse

class EventService:
    """
    业务逻辑层 (Service Layer)
    负责：
    1. 接收控制器传输的请求 (Pydantic Model)
    2. 调用 Client 获取原始数据
    3. 处理/清洗数据 (例如标准化日期、格式化字段)
    4. 返回标准化的 Event 模型给控制器
    """

    def __init__(self, tavily_client: TavilyClient):
        # 依赖注入 Client
        self.tavily_client = tavily_client

    async def fetch_news_events(self, request: EventSearchRequest) -> EventSearchResponse:
        """
        根据搜索请求获取新闻事件
        Args:
            request (EventSearchRequest): 包含 query, start_date, end_date 的请求对象
            
        Returns:
            EventSearchResponse: 包含 EventItem 列表
        """
        # 转换日期格式 if needed (Tavily api accepts strings, but Pydantic gives date objects)
        start_str = request.start_date.strftime("%Y-%m-%d") if request.start_date else None
        end_str = request.end_date.strftime("%Y-%m-%d") if request.end_date else None
        
        # 调用 client 获取原始数据
        # client 的 search 方法做了异步请求
        raw_results = await self.tavily_client.search(
            query=request.query,
            start_date=start_str,
            end_date=end_str
        )
        
        # 处理数据：从 raw dict 提取 title, url, content, published_date
        event_items: List[EventItem] = []
        for result in raw_results:
            # 安全获取字段，处理缺失值
            title = result.get("title", "No Title")
            url = result.get("url", "#")
            content = result.get("content") or result.get("snippet", "No Content available")
            
            # published_date 处理 (API 返回的日期格式可能不统一，这里简单处理)
            # Tavily 可能会有 'published_date': '2023-10-25T...'
            # 可以加入日期解析逻辑，这里保持原样或提供默认值
            published_date = result.get("published_date")
            
            item = EventItem(
                title=title,
                url=url,
                content=content, 
                published_date=published_date
            )
            event_items.append(item)
            
        # 返回标准的响应模型
        return EventSearchResponse(
            total_results=len(event_items),
            results=event_items
        )
