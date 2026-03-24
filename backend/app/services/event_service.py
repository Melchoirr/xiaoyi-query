from bisect import bisect_left
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.clients.polymarket_client import PolymarketClient
from app.clients.tavily_client import TavilyClient
from app.schemas.event import (
    AlignedEventPoint,
    EventAlignmentRequest,
    EventAlignmentResponse,
    EventItem,
    EventSearchRequest,
    EventSearchResponse,
    MarketTimePoint,
    PolymarketEventSummary,
)

class EventService:
    """
    业务逻辑层 (Service Layer)
    负责：
    1. 接收控制器传输的请求 (Pydantic Model)
    2. 调用 Client 获取原始数据
    3. 处理/清洗数据 (例如标准化日期、格式化字段)
    4. 返回标准化的 Event 模型给控制器
    """

    def __init__(self, tavily_client: TavilyClient, polymarket_client: PolymarketClient):
        # 依赖注入 Client
        self.tavily_client = tavily_client
        self.polymarket_client = polymarket_client

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

    async def fetch_event_alignment(self, request: EventAlignmentRequest) -> EventAlignmentResponse:
        """
        搜索事件并返回 Polymarket 概率随时间变化，以及新闻对齐结果。
        """
        # 1) 新闻
        news_request = EventSearchRequest(
            query=request.query,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        news_response = await self.fetch_news_events(news_request)
        news_items = news_response.results[: request.news_limit]

        # 2) Polymarket 事件候选
        event_candidates = await self.polymarket_client.search_events(
            query=request.query,
            limit=request.event_limit,
            active=True,
            closed=False,
        )

        if not event_candidates:
            return EventAlignmentResponse(
                query=request.query,
                news=news_items,
                note="No matching Polymarket events were found.",
            )

        selected_event = self._select_best_event(event_candidates, request.query)
        market = self._select_market_with_token(selected_event)
        if market is None:
            return EventAlignmentResponse(
                query=request.query,
                news=news_items,
                note="A matching event was found, but no market token is available for price history.",
            )

        token_id = self.polymarket_client.extract_token_id(market)
        if not token_id:
            return EventAlignmentResponse(
                query=request.query,
                news=news_items,
                note="A matching market was found, but token parsing failed.",
            )

        start_ts = self.polymarket_client.date_to_ts(request.start_date) if request.start_date else None
        end_ts = self.polymarket_client.date_to_ts(request.end_date) if request.end_date else None

        # 3) 价格历史
        raw_history = await self.polymarket_client.get_price_history(
            token_id=token_id,
            start_ts=start_ts,
            end_ts=end_ts,
            fidelity=request.fidelity,
        )

        market_series = self._to_market_points(raw_history)

        # 4) 对齐：将新闻时间戳映射到最近 market 点
        aligned_events = self._align_news_to_market(news_items, market_series)

        summary = PolymarketEventSummary(
            id=str(selected_event.get("id", "")),
            title=str(selected_event.get("title", "")),
            slug=selected_event.get("slug"),
            category=selected_event.get("category"),
            market_id=str(market.get("id")) if market.get("id") is not None else None,
            market_question=market.get("question"),
            token_id=token_id,
        )

        return EventAlignmentResponse(
            query=request.query,
            selected_event=summary,
            market_series=market_series,
            news=news_items,
            aligned_events=aligned_events,
            note=None if market_series else "Price history is empty for the selected market/token.",
        )

    def _to_market_points(self, history: List[Dict[str, Any]]) -> List[MarketTimePoint]:
        points: List[MarketTimePoint] = []
        for item in history:
            ts = item.get("t")
            price = item.get("p")
            if ts is None or price is None:
                continue
            try:
                ts_int = int(ts)
                price_float = float(price)
            except (TypeError, ValueError):
                continue

            dt_iso = datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat()
            points.append(MarketTimePoint(timestamp=ts_int, datetime=dt_iso, price=price_float))

        points.sort(key=lambda x: x.timestamp)
        return points

    def _align_news_to_market(
        self,
        news_items: List[EventItem],
        market_series: List[MarketTimePoint],
    ) -> List[AlignedEventPoint]:
        if not market_series:
            return [
                AlignedEventPoint(
                    title=item.title,
                    url=item.url,
                    news_time=item.published_date,
                )
                for item in news_items
            ]

        market_timestamps = [p.timestamp for p in market_series]
        aligned: List[AlignedEventPoint] = []

        for item in news_items:
            news_ts = self._parse_datetime_to_ts(item.published_date)
            if news_ts is None:
                aligned.append(
                    AlignedEventPoint(
                        title=item.title,
                        url=item.url,
                        news_time=item.published_date,
                    )
                )
                continue

            idx = bisect_left(market_timestamps, news_ts)
            candidates = []
            if 0 <= idx < len(market_series):
                candidates.append(market_series[idx])
            if idx - 1 >= 0:
                candidates.append(market_series[idx - 1])

            nearest = min(candidates, key=lambda p: abs(p.timestamp - news_ts)) if candidates else None
            aligned.append(
                AlignedEventPoint(
                    title=item.title,
                    url=item.url,
                    news_time=item.published_date,
                    market_timestamp=nearest.timestamp if nearest else None,
                    market_datetime=nearest.datetime if nearest else None,
                    market_price=nearest.price if nearest else None,
                )
            )

        return aligned

    def _select_market_with_token(self, event_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        markets = event_item.get("markets", [])
        if not isinstance(markets, list):
            return None

        for market in markets:
            token = self.polymarket_client.extract_token_id(market)
            if token:
                return market
        return None

    def _select_best_event(self, candidates: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        q = query.strip().lower()
        tokens = [t for t in q.split() if t]

        def score(item: Dict[str, Any]) -> int:
            title = str(item.get("title", "")).lower()
            slug = str(item.get("slug", "")).lower()
            text = f"{title} {slug}"

            s = 0
            if q and q in text:
                s += 5
            for token in tokens:
                if token in text:
                    s += 1
            if item.get("active"):
                s += 1
            if not item.get("closed"):
                s += 1
            return s

        return max(candidates, key=score)

    def _parse_datetime_to_ts(self, value: Optional[str]) -> Optional[int]:
        if not value:
            return None

        s = value.strip()
        if not s:
            return None

        # Tavily 常见返回：ISO8601，可能包含 Z
        iso_value = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(iso_value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            return None
