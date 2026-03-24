from typing import List, Optional

from fastapi import HTTPException

from app.clients.polymarket_client import PolymarketClient
from app.clients.tavily_client import TavilyClient
from app.schemas.event import (
    EventAlignmentRequest,
    EventAlignmentResponse,
    EventSearchRequest,
    EventSearchResponse,
    PolymarketEventSummary,
    PriceSpikeAlert,
)
from app.services.event_alignment_utils import align_news_to_market, detect_price_spikes
from app.services.event_market import build_alignment_note, pick_market_with_history, to_market_points
from app.services.event_matching import select_best_event
from app.services.event_news import fetch_news_events as fetch_news_events_impl
from app.services.event_news import query_news_for_spike


class EventService:
    """
    业务逻辑层（编排器）。

    拆分后职责：
    - 保持对外接口不变（供 router 调用）
    - 负责流程编排与错误处理
    - 将细节逻辑委托给 app/services 下的子模块
    """

    def __init__(self, tavily_client: TavilyClient, polymarket_client: PolymarketClient):
        self.tavily_client = tavily_client
        self.polymarket_client = polymarket_client

    async def fetch_news_events(self, request: EventSearchRequest) -> EventSearchResponse:
        """根据搜索请求获取新闻事件。"""
        return await fetch_news_events_impl(self.tavily_client, request)

    async def fetch_event_alignment(self, request: EventAlignmentRequest) -> EventAlignmentResponse:
        """搜索事件并返回 Polymarket 概率随时间变化，以及新闻对齐结果。"""
        notes: List[str] = []

        news_request = EventSearchRequest(
            query=request.query,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        try:
            news_response = await self.fetch_news_events(news_request)
            news_items = news_response.results[: request.news_limit]
        except HTTPException as exc:
            news_items = []
            notes.append(f"News provider unavailable ({exc.status_code}): {exc.detail}")

        try:
            event_candidates = await self.polymarket_client.search_events(
                query=request.query,
                limit=request.event_limit,
                active=True,
                closed=False,
            )
        except HTTPException as exc:
            notes.append(f"Polymarket unavailable ({exc.status_code}): {exc.detail}")
            return EventAlignmentResponse(
                query=request.query,
                news=news_items,
                note="; ".join(notes),
            )

        if not event_candidates:
            return EventAlignmentResponse(
                query=request.query,
                news=news_items,
                note=self._merge_note(notes, "No matching Polymarket events were found."),
            )

        selected_event = select_best_event(event_candidates, request.query)
        if selected_event is None:
            return EventAlignmentResponse(
                query=request.query,
                news=news_items,
                note=self._merge_note(notes, "No relevant Polymarket event matched the query."),
            )

        start_ts = self.polymarket_client.date_to_ts(request.start_date) if request.start_date else None
        end_ts = self.polymarket_client.date_to_ts(request.end_date) if request.end_date else None

        market, token_id, raw_history, history_fallback_used = await pick_market_with_history(
            polymarket_client=self.polymarket_client,
            event_item=selected_event,
            start_ts=start_ts,
            end_ts=end_ts,
            fidelity=request.fidelity,
        )

        if market is None or token_id is None:
            return EventAlignmentResponse(
                query=request.query,
                news=news_items,
                note=self._merge_note(
                    notes,
                    "A matching event was found, but no market token is available for price history.",
                ),
            )

        market_series = to_market_points(raw_history)
        aligned_events = align_news_to_market(news_items, market_series)

        price_spike_alerts: List[PriceSpikeAlert] = []
        if request.detect_spikes and market_series:
            spike_points = detect_price_spikes(market_series, request.price_spike_threshold)
            for spike_point in spike_points:
                spike_news = await query_news_for_spike(
                    tavily_client=self.tavily_client,
                    query=request.query,
                    spike_timestamp=spike_point.timestamp,
                    time_window_hours=24,
                )
                price_spike_alerts.append(
                    PriceSpikeAlert(
                        spike=spike_point,
                        related_news=spike_news,
                    )
                )

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
            price_spike_alerts=price_spike_alerts,
            note=self._merge_note(
                notes,
                build_alignment_note(
                    market_series=market_series,
                    history_fallback_used=history_fallback_used,
                ),
            ),
        )

    def _merge_note(self, notes: List[str], final_note: Optional[str]) -> Optional[str]:
        merged = [n for n in notes if n]
        if final_note:
            merged.append(final_note)
        return "; ".join(merged) if merged else None
