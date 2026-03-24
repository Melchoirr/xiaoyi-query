from datetime import datetime, timezone
from typing import List

from fastapi import HTTPException

from app.clients.tavily_client import TavilyClient
from app.schemas.event import EventItem, EventSearchRequest, EventSearchResponse


def build_event_item(result: dict) -> EventItem:
    """Normalize a Tavily raw result into EventItem."""
    title = result.get("title", "No Title")
    url = result.get("url", "#")
    content = result.get("content") or result.get("snippet", "No Content available")
    published_date = result.get("published_date")
    return EventItem(
        title=title,
        url=url,
        content=content,
        published_date=published_date,
    )


async def fetch_news_events(
    tavily_client: TavilyClient,
    request: EventSearchRequest,
) -> EventSearchResponse:
    """Fetch and normalize news events from Tavily."""
    start_str = request.start_date.strftime("%Y-%m-%d") if request.start_date else None
    end_str = request.end_date.strftime("%Y-%m-%d") if request.end_date else None

    raw_results = await tavily_client.search(
        query=request.query,
        start_date=start_str,
        end_date=end_str,
    )

    event_items: List[EventItem] = [build_event_item(result) for result in raw_results]
    return EventSearchResponse(total_results=len(event_items), results=event_items)


async def query_news_for_spike(
    tavily_client: TavilyClient,
    query: str,
    spike_timestamp: int,
    time_window_hours: int = 24,
) -> List[EventItem]:
    """Search related news in a time window around a spike timestamp."""
    half_window_sec = (time_window_hours * 3600) // 2
    start_ts = spike_timestamp - half_window_sec
    end_ts = spike_timestamp + half_window_sec

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    try:
        raw_results = await tavily_client.search(
            query=query,
            start_date=start_date_str,
            end_date=end_date_str,
        )
    except HTTPException:
        return []

    return [build_event_item(result) for result in raw_results]
