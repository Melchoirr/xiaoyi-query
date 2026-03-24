from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.clients.polymarket_client import PolymarketClient
from app.schemas.event import MarketTimePoint


async def pick_market_with_history(
    polymarket_client: PolymarketClient,
    event_item: Dict[str, Any],
    start_ts: Optional[int],
    end_ts: Optional[int],
    fidelity: int,
) -> tuple[Optional[Dict[str, Any]], Optional[str], List[Dict[str, Any]], bool]:
    """Select a market/token with available price history.

    Returns: (market, token_id, history, fallback_used)
    """
    markets = event_item.get("markets", [])
    if not isinstance(markets, list) or not markets:
        return None, None, [], False

    token_markets: List[tuple[Dict[str, Any], str]] = []
    for market in markets:
        token = polymarket_client.extract_token_id(market)
        if token:
            token_markets.append((market, token))

    if not token_markets:
        return None, None, [], False

    for market, token in token_markets:
        try:
            history = await polymarket_client.get_price_history(
                token_id=token,
                start_ts=start_ts,
                end_ts=end_ts,
                fidelity=fidelity,
            )
        except HTTPException:
            continue
        if history:
            return market, token, history, False

    if start_ts is not None or end_ts is not None:
        for market, token in token_markets:
            try:
                history = await polymarket_client.get_price_history(
                    token_id=token,
                    start_ts=None,
                    end_ts=None,
                    fidelity=fidelity,
                )
            except HTTPException:
                continue
            if history:
                return market, token, history, True

    market, token = token_markets[0]
    return market, token, [], False


def build_alignment_note(market_series: List[MarketTimePoint], history_fallback_used: bool) -> Optional[str]:
    """Build a user-facing note for history availability."""
    if market_series and not history_fallback_used:
        return None
    if market_series and history_fallback_used:
        return "No price points were found in the requested date range; returned available full history instead."
    return "Price history is empty for the selected market/token."


def to_market_points(history: List[Dict[str, Any]]) -> List[MarketTimePoint]:
    """Convert raw price history into sorted MarketTimePoint list."""
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
