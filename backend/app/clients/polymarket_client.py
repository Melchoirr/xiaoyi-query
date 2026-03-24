import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException


class PolymarketClient:
    """
    Polymarket 数据访问客户端。

    职责：
    1. 调用 Gamma API 获取事件列表（events）
    2. 调用 CLOB API 获取价格时间序列（prices-history）
    3. 对外部接口返回做最小标准化，统一异常为 HTTPException

    说明：
    - 该类只负责“数据访问”，不处理业务规则（例如事件打分、对齐逻辑）。
    - 业务层（Service）可在此基础上做进一步筛选和回退策略。
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
        """
        获取 Polymarket 事件候选列表，并在本地进行关键词匹配。

        Args:
            query: 用户输入的查询词。
            limit: 返回结果上限。
            active: 是否只查询活跃事件。
            closed: 是否包含已关闭事件。

        Returns:
            事件字典列表。若本地匹配命中，则返回命中结果；否则返回前 limit 条原始候选。

        Raises:
            HTTPException: 当上游 API 返回非 2xx 或网络不可达时抛出。
        """
        # Gamma 的 /events 接口对关键词过滤支持有限，适当扩大候选池后做本地相关性筛选。
        fetch_limit = max(limit, 200)
        params = {
            "limit": fetch_limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                # 先获取较大候选集，再在本地做关键词匹配，提升召回率。
                response = await client.get(
                    f"{self.GAMMA_BASE_URL}/events",
                    params=params,
                    headers=headers,
                    timeout=12.0,
                )
                response.raise_for_status()
                events = response.json()
            except httpx.HTTPStatusError as e:
                # 上游明确返回错误状态码（4xx/5xx）。
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail="Failed to fetch events from Polymarket Gamma API",
                )
            except httpx.RequestError:
                # 网络错误、超时、DNS 解析失败等请求级异常。
                raise HTTPException(
                    status_code=503,
                    detail="Service unavailable: Could not connect to Polymarket Gamma API",
                )

        query_lower = query.strip().lower()
        if not query_lower:
            return events

        def _match(ev: Dict[str, Any]) -> bool:
            """本地匹配：在 title/slug/description/market question 中查找 query。"""
            title = str(ev.get("title", "")).lower()
            slug = str(ev.get("slug", "")).lower()
            description = str(ev.get("description", "")).lower()

            questions: List[str] = []
            markets = ev.get("markets", [])
            if isinstance(markets, list):
                for market in markets:
                    q = market.get("question")
                    if q:
                        questions.append(str(q).lower())

            text = " ".join([title, slug, description] + questions)
            return query_lower in text

        # 若匹配到相关事件，优先返回匹配集合；否则回退到原始候选前 limit 条。
        matched = [ev for ev in events if _match(ev)]
        return matched[:limit] if matched else events[:limit]

    async def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        fidelity: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        获取指定 token 的价格历史序列。

        Args:
            token_id: CLOB 市场 token id。
            start_ts: 起始 Unix 时间戳（秒），可选。
            end_ts: 结束 Unix 时间戳（秒），可选。
            fidelity: 采样粒度（秒级步长，具体支持值由上游决定）。

        Returns:
            历史点列表（形如 {"t": ts, "p": price}）。
            若上游返回结构缺失或 history 非列表，则返回空列表。

        Raises:
            HTTPException: 当上游 API 返回非 2xx 或网络不可达时抛出。
        """
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
                # CLOB prices-history 接口按 market(token) 查询时间序列。
                response = await client.get(
                    f"{self.CLOB_BASE_URL}/prices-history",
                    params=params,
                    headers=headers,
                    timeout=12.0,
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPStatusError as e:
                # 上游返回明确的状态码（常见如 400/404）。
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail="Failed to fetch price history from Polymarket CLOB API",
                )
            except httpx.RequestError:
                # 网络不可达或请求超时等场景统一为 503。
                raise HTTPException(
                    status_code=503,
                    detail="Service unavailable: Could not connect to Polymarket CLOB API",
                )

        history = payload.get("history", [])
        # 兜底保证返回类型稳定，避免上层出现类型判断分支膨胀。
        return history if isinstance(history, list) else []

    @staticmethod
    def extract_token_id(market: Dict[str, Any]) -> Optional[str]:
        """
        解析 market 中的 clobTokenIds，优先返回 Yes token。

        Args:
            market: 单个 market 字典，通常来自 Gamma event.markets。

        Returns:
            可用 token id（字符串）；若无法解析则返回 None。

        兼容格式：
        - list: ["tokenA", "tokenB"]
        - json string: "[\"tokenA\", \"tokenB\"]"
        - plain string: "tokenA"
        """
        raw_token_ids = market.get("clobTokenIds")
        if raw_token_ids is None:
            return None

        token_ids: List[str] = []
        if isinstance(raw_token_ids, list):
            token_ids = [str(x) for x in raw_token_ids]
        elif isinstance(raw_token_ids, str):
            try:
                # 一些字段会以 JSON 字符串形式返回列表。
                parsed = json.loads(raw_token_ids)
                if isinstance(parsed, list):
                    token_ids = [str(x) for x in parsed]
                else:
                    # JSON 解析成功但不是 list，按单值字符串处理。
                    token_ids = [raw_token_ids]
            except json.JSONDecodeError:
                # 不是合法 JSON，视为普通 token 字符串。
                token_ids = [raw_token_ids]

        # 当前实现返回首个 token；更精细的 Yes/No 选择可在后续按市场结构增强。
        return token_ids[0] if token_ids else None

    @staticmethod
    def date_to_ts(date_obj) -> int:
        """
        将 date 对象转换为 UTC 当日 00:00:00 的 Unix 时间戳（秒）。

        Args:
            date_obj: datetime.date 对象。

        Returns:
            UTC 秒级时间戳。
        """
        # 将 date 视作 UTC 当天零点，避免时区歧义。
        dt = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
