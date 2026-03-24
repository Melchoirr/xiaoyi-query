import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException


class PolymarketClient:
    """
    封装 Polymarket Gamma/CLOB API 调用。
    """

    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
    CLOB_BASE_URL = "https://clob.polymarket.com"

    async def search_events(
        self,
        query: str,
        limit: int = 20,
        active: bool = True,
        closed: bool = False,
    ) -> List[Dict[str, Any]]:
        params = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.GAMMA_BASE_URL}/events",
                    params=params,
                    headers=headers,
                    timeout=12.0,
                )
                response.raise_for_status()
                events = response.json()
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail="Failed to fetch events from Polymarket Gamma API",
                )
            except httpx.RequestError:
                raise HTTPException(
                    status_code=503,
                    detail="Service unavailable: Could not connect to Polymarket Gamma API",
                )

        query_lower = query.strip().lower()
        if not query_lower:
            return events

        def _match(ev: Dict[str, Any]) -> bool:
            title = str(ev.get("title", "")).lower()
            slug = str(ev.get("slug", "")).lower()
            return query_lower in title or query_lower in slug

        matched = [ev for ev in events if _match(ev)]
        return matched or events

    async def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        fidelity: int = 60,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "market": token_id,
            "interval": "max",
            "fidelity": fidelity,
        }
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.CLOB_BASE_URL}/prices-history",
                    params=params,
                    headers=headers,
                    timeout=12.0,
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail="Failed to fetch price history from Polymarket CLOB API",
                )
            except httpx.RequestError:
                raise HTTPException(
                    status_code=503,
                    detail="Service unavailable: Could not connect to Polymarket CLOB API",
                )

        history = payload.get("history", [])
        return history if isinstance(history, list) else []

    @staticmethod
    def extract_token_id(market: Dict[str, Any]) -> Optional[str]:
        """
        解析 market 中的 clobTokenIds，优先返回 Yes token。
        """
        raw_token_ids = market.get("clobTokenIds")
        if raw_token_ids is None:
            return None

        token_ids: List[str] = []
        if isinstance(raw_token_ids, list):
            token_ids = [str(x) for x in raw_token_ids]
        elif isinstance(raw_token_ids, str):
            try:
                parsed = json.loads(raw_token_ids)
                if isinstance(parsed, list):
                    token_ids = [str(x) for x in parsed]
                else:
                    token_ids = [raw_token_ids]
            except json.JSONDecodeError:
                token_ids = [raw_token_ids]

        return token_ids[0] if token_ids else None

    @staticmethod
    def date_to_ts(date_obj) -> int:
        dt = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
