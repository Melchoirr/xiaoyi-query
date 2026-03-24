from bisect import bisect_left
from datetime import datetime, timezone
from typing import List, Optional

from app.schemas.event import AlignedEventPoint, EventItem, MarketTimePoint, PriceSpikePoint


def parse_datetime_to_ts(value: Optional[str]) -> Optional[int]:
    """Parse ISO datetime string into unix timestamp (seconds)."""
    if not value:
        return None

    s = value.strip()
    if not s:
        return None

    iso_value = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


def align_news_to_market(
    news_items: List[EventItem],
    market_series: List[MarketTimePoint],
) -> List[AlignedEventPoint]:
    """Align each news item to nearest market point by timestamp."""
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
        news_ts = parse_datetime_to_ts(item.published_date)
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


def detect_price_spikes(market_series: List[MarketTimePoint], threshold: float) -> List[PriceSpikePoint]:
    """Detect adjacent-point price spikes over absolute threshold."""
    spikes: List[PriceSpikePoint] = []

    if len(market_series) < 2:
        return spikes

    for i in range(1, len(market_series)):
        prev_point = market_series[i - 1]
        curr_point = market_series[i]

        price_change = abs(curr_point.price - prev_point.price)
        price_change_pct = price_change / prev_point.price if prev_point.price != 0 else 0

        if price_change >= threshold:
            spikes.append(
                PriceSpikePoint(
                    timestamp=curr_point.timestamp,
                    datetime=curr_point.datetime,
                    price_before=prev_point.price,
                    price_after=curr_point.price,
                    price_change=price_change,
                    price_change_pct=price_change_pct,
                )
            )

    return spikes
