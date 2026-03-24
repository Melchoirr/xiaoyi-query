from typing import Any, Dict, List, Optional


def build_query_terms(query: str) -> List[str]:
    """Build normalized query terms with Chinese financial aliases."""
    raw = query.strip().lower()
    if not raw:
        return []

    alias_map = {
        "美联储": ["fed", "federal reserve", "fomc"],
        "加息": ["rate hike", "interest rate"],
        "降息": ["rate cut", "interest rate"],
        "通胀": ["inflation", "cpi"],
        "比特币": ["bitcoin", "btc"],
        "以太坊": ["ethereum", "eth"],
        "特朗普": ["trump"],
        "拜登": ["biden"],
    }

    terms: List[str] = []
    normalized = (
        raw.replace("？", " ")
        .replace("?", " ")
        .replace("，", " ")
        .replace(",", " ")
        .replace("。", " ")
        .replace("!", " ")
        .replace("！", " ")
    )

    for token in normalized.split():
        if token and token not in terms:
            terms.append(token)

    for zh, aliases in alias_map.items():
        if zh in raw:
            for alias in aliases:
                if alias not in terms:
                    terms.append(alias)

    return terms


def select_best_event(candidates: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    """Score and select the most relevant event candidate."""
    terms = build_query_terms(query)

    def score(item: Dict[str, Any]) -> int:
        title = str(item.get("title", "")).lower()
        slug = str(item.get("slug", "")).lower()

        questions: List[str] = []
        markets = item.get("markets", [])
        if isinstance(markets, list):
            for market in markets:
                qn = market.get("question")
                if qn:
                    questions.append(str(qn).lower())

        text = " ".join([title, slug] + questions)

        s = 0
        for term in terms:
            if term in title or term in slug:
                s += 3
            elif term in text:
                s += 1
        return s

    if not candidates:
        return None

    scored = [(score(item), item) for item in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_item = scored[0]
    if best_score <= 0:
        return None
    return best_item
