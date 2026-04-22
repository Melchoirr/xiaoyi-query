import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import requests


DB_PATH = "data/gdelt_master.duckdb"

# Keep window aligned with existing experiments for comparability.
START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"

SEED_QUERIES = [
    "Trump", "Biden", "election", "Fed", "interest rates", "inflation",
    "recession", "tariff", "China", "Taiwan", "Ukraine", "Russia",
    "Israel", "Iran", "oil", "Bitcoin", "Ethereum", "ETF", "Tesla",
    "Nvidia", "OpenAI", "ChatGPT", "AI", "SpaceX", "climate",
]

TECH_SEED_QUERIES = [
    "AI", "OpenAI", "ChatGPT", "Nvidia", "Tesla", "Apple", "Google", "Meta",
    "Microsoft", "semiconductor", "chip", "GPU", "SpaceX", "Bitcoin", "Ethereum", "ETF",
]

STOPWORDS = {
    "will", "the", "a", "an", "and", "or", "for", "to", "in", "on",
    "of", "is", "are", "be", "by", "with", "from", "after", "before",
    "this", "that", "at", "as", "if", "it", "than", "new", "over",
    "under", "into", "out", "who", "what", "when", "where", "why", "how",
}

LIQUIDITY_KEYS = [
    "volumeNum", "volume", "volume24hr", "volume24hrClob", "oneDayVolume",
    "liquidityNum", "liquidity", "liquidityClob",
]

# Heuristics for title-only retrieval suitability.
CORE_ENTITY_HINTS = {
    "trump", "biden", "fed", "fomc", "powell", "us", "u.s", "usa", "china", "taiwan",
    "iran", "israel", "russia", "ukraine", "opec", "oil", "crude", "bitcoin", "ethereum",
    "etf", "tesla", "nvidia", "openai", "chatgpt", "greenland",
}

STRONG_ACTION_HINTS = {
    "decision", "decrease", "increase", "cut", "hike", "invade", "acquire", "nominate",
    "approve", "ban", "ceasefire", "strike", "sanction", "winner", "election", "conflict",
}

AMBIGUOUS_ACTION_HINTS = {
    "visit", "leadership", "support", "announce", "change", "speech", "meeting",
}

EXCLUDED_TITLE_PATTERNS = [
    r"\bpresidential\s+election\b",
    r"\belection\s+winner\b",
    r"\bwin\s+the\s+\d{4}\s+us\s+presidential\s+election\b",
]

TECH_TITLE_HINTS = {
    "ai", "openai", "chatgpt", "nvidia", "tesla", "apple", "google", "meta", "microsoft",
    "semiconductor", "chip", "gpu", "spacex", "bitcoin", "ethereum", "etf", "crypto", "robot",
}

POLITICAL_TITLE_HINTS = {
    "trump", "biden", "fed", "fomc", "election", "presidential", "iran", "israel", "ukraine",
    "russia", "ceasefire", "invade", "tariff", "nuclear", "gulf", "taiwan", "greenland",
}


@dataclass
class EventCandidate:
    event_id: str
    event_title: str
    event_slug: str
    market_count: int
    liquidity_score_raw: float
    news_count: int
    news_keywords: str
    title_fit_score: float
    title_fit_note: str
    combined_score: float


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def tokenize_title(title: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']+", title.lower())
    tokens = [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]

    # Keep order while deduplicating.
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:4]


def choose_news_keywords(event_title: str) -> list[str]:
    toks = tokenize_title(event_title)
    if len(toks) >= 2:
        return toks[:2]
    if len(toks) == 1:
        return toks

    # Fallback to a coarse token if title parsing fails.
    words = [w for w in re.split(r"\W+", event_title) if w]
    return [words[0].lower()] if words else ["market"]


def score_title_fit(event_title: str) -> tuple[float, str]:
    title_l = (event_title or "").lower()
    toks = set(tokenize_title(event_title))

    entity_hits = sum(1 for h in CORE_ENTITY_HINTS if re.search(rf"\b{re.escape(h)}\b", title_l))
    action_hits = sum(1 for h in STRONG_ACTION_HINTS if re.search(rf"\b{re.escape(h)}\b", title_l))
    ambiguous_hits = sum(1 for h in AMBIGUOUS_ACTION_HINTS if re.search(rf"\b{re.escape(h)}\b", title_l))

    geo_hits = 0
    for g in ["iran", "israel", "us", "u.s", "china", "russia", "ukraine", "taiwan"]:
        if re.search(rf"\b{re.escape(g)}\b", title_l):
            geo_hits += 1

    entity_part = min(1.0, entity_hits / 2.0)
    action_part = min(1.0, action_hits / 1.0)
    geo_pair_bonus = 0.2 if geo_hits >= 2 else 0.0
    token_bonus = 0.1 if len(toks) >= 2 else 0.0
    ambiguity_penalty = 0.15 if ambiguous_hits > 0 and action_hits == 0 else 0.0

    score = 0.45 * entity_part + 0.35 * action_part + geo_pair_bonus + token_bonus - ambiguity_penalty
    score = max(0.0, min(1.0, score))

    note = (
        f"entity_hits={entity_hits}; action_hits={action_hits}; "
        f"geo_hits={geo_hits}; ambiguous_hits={ambiguous_hits}"
    )
    return score, note


def is_excluded_event_title(event_title: str) -> bool:
    title_l = (event_title or "").lower()
    return any(re.search(pat, title_l) for pat in EXCLUDED_TITLE_PATTERNS)


def is_tech_title(event_title: str) -> bool:
    title_l = (event_title or "").lower()
    return any(re.search(rf"\b{re.escape(h)}\b", title_l) for h in TECH_TITLE_HINTS)


def is_political_title(event_title: str) -> bool:
    title_l = (event_title or "").lower()
    return any(re.search(rf"\b{re.escape(h)}\b", title_l) for h in POLITICAL_TITLE_HINTS)


def fetch_candidates_from_polymarket(seed_queries: list[str]) -> dict[str, dict]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }
    )

    by_id: dict[str, dict] = {}

    for query in seed_queries:
        resp = session.get(
            "https://gamma-api.polymarket.com/public-search",
            params={"q": query, "limit_per_type": 40, "keep_closed_markets": 1},
            timeout=20,
        )
        resp.raise_for_status()
        events = resp.json().get("events", [])

        start_window = pd.to_datetime(START_DATE, utc=True)
        end_window = pd.to_datetime(END_DATE, utc=True)

        for ev in events:
            eid = str(ev.get("id", ""))
            if not eid:
                continue

            markets = ev.get("markets", []) or []
            raw_liq = 0.0
            overlap_market_count = 0
            for m in markets:
                m_start = pd.to_datetime(m.get("startDate"), utc=True, errors="coerce")
                m_end = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")

                if pd.isna(m_start) and pd.isna(m_end):
                    overlaps = True
                elif pd.isna(m_start):
                    overlaps = m_end >= start_window
                elif pd.isna(m_end):
                    overlaps = m_start <= end_window
                else:
                    overlaps = (m_start <= end_window) and (m_end >= start_window)

                if not overlaps:
                    continue

                m_values = [to_float(m.get(k)) for k in LIQUIDITY_KEYS]
                raw_liq += max(m_values) if m_values else 0.0
                overlap_market_count += 1

            if overlap_market_count == 0:
                continue

            prev = by_id.get(eid)
            candidate = {
                "event_id": eid,
                "event_title": ev.get("title", ""),
                "event_slug": ev.get("slug", ""),
                "market_count": overlap_market_count,
                "liquidity_score_raw": raw_liq,
            }
            if (prev is None) or (candidate["liquidity_score_raw"] > prev["liquidity_score_raw"]):
                by_id[eid] = candidate

    return by_id


def count_news_for_event(con: duckdb.DuckDBPyConnection, title: str, start_ts: str, end_ts: str) -> tuple[int, str]:
    keywords = choose_news_keywords(title)
    clauses = []
    for kw in keywords:
        safe = kw.replace("'", "''")
        clauses.append(f"Estimated_Title ILIKE '%{safe}%' ")

    where_kw = " OR ".join(clauses) if clauses else "1=0"
    start_key = start_ts.replace("-", "").replace(" ", "").replace(":", "")
    end_key = end_ts.replace("-", "").replace(" ", "").replace(":", "")

    sql = f"""
    SELECT COUNT(*) AS n
    FROM gkg_news
    WHERE CAST(DATE AS VARCHAR) >= '{start_key}'
      AND CAST(DATE AS VARCHAR) <= '{end_key}'
      AND ({where_kw})
    """
    n = int(con.execute(sql).fetchone()[0])
    return n, ",".join(keywords)


def build_event_pool(top_k: int = 10, min_title_fit: float = 0.45, theme: str = "general") -> pd.DataFrame:
    seed_queries = TECH_SEED_QUERIES if theme == "tech" else SEED_QUERIES
    candidates = fetch_candidates_from_polymarket(seed_queries=seed_queries)
    if not candidates:
        raise RuntimeError("No event candidates fetched from Polymarket public-search.")

    con = duckdb.connect(DB_PATH, read_only=True)
    rows: list[EventCandidate] = []

    for ev in candidates.values():
        if theme == "tech":
            if (not is_tech_title(ev["event_title"])) or is_political_title(ev["event_title"]):
                continue

        if is_excluded_event_title(ev["event_title"]):
            continue

        news_count, news_keywords = count_news_for_event(
            con,
            title=ev["event_title"],
            start_ts=START_DATE,
            end_ts=END_DATE,
        )
        title_fit_score, title_fit_note = score_title_fit(ev["event_title"])

        rows.append(
            EventCandidate(
                event_id=ev["event_id"],
                event_title=ev["event_title"],
                event_slug=ev["event_slug"],
                market_count=ev["market_count"],
                liquidity_score_raw=ev["liquidity_score_raw"],
                news_count=news_count,
                news_keywords=news_keywords,
                title_fit_score=title_fit_score,
                title_fit_note=title_fit_note,
                combined_score=0.0,
            )
        )

    con.close()

    df = pd.DataFrame([r.__dict__ for r in rows])

    # Filter out obviously unusable events first.
    df = df[(df["liquidity_score_raw"] > 0) & (df["news_count"] > 0)].copy()
    df = df[df["title_fit_score"] >= float(min_title_fit)].copy()
    if df.empty:
        raise RuntimeError("No event meets liquidity/news coverage and title-fit threshold.")

    liq = df["liquidity_score_raw"].apply(lambda x: math.log1p(max(x, 0.0)))
    news = df["news_count"].apply(lambda x: math.log1p(max(float(x), 0.0)))

    liq_n = (liq - liq.min()) / (liq.max() - liq.min() + 1e-9)
    news_n = (news - news.min()) / (news.max() - news.min() + 1e-9)

    # Prioritize title-news compatibility, then liquidity/news scale.
    fit_n = df["title_fit_score"].clip(lower=0.0, upper=1.0)
    df["combined_score"] = 0.5 * fit_n + 0.3 * liq_n + 0.2 * news_n

    # Keep diversity by removing near-duplicate titles.
    df = df.sort_values("combined_score", ascending=False).copy()
    keep_rows = []
    seen_signatures = set()
    for _, row in df.iterrows():
        signature = " ".join(tokenize_title(row["event_title"])[:3])
        if signature and signature in seen_signatures:
            continue
        if signature:
            seen_signatures.add(signature)
        keep_rows.append(row)
        if len(keep_rows) >= top_k:
            break

    out = pd.DataFrame(keep_rows)
    return out[
        [
            "event_id",
            "event_title",
            "event_slug",
            "market_count",
            "liquidity_score_raw",
            "news_count",
            "news_keywords",
            "title_fit_score",
            "title_fit_note",
            "combined_score",
        ]
    ].reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build title-news compatible event pool")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-title-fit", type=float, default=0.45)
    parser.add_argument("--theme", choices=["general", "tech"], default="general")
    args = parser.parse_args()

    today = date.today().isoformat()
    out_dir = Path("result") / "by_date" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_tech" if args.theme == "tech" else ""
    out_csv = out_dir / f"event_pool_top10_{today}{suffix}.csv"
    out_md = out_dir / f"event_pool_top10_{today}{suffix}.md"

    df = build_event_pool(top_k=args.top_k, min_title_fit=args.min_title_fit, theme=args.theme)
    df.to_csv(out_csv, index=False)

    lines = [
        "# High-Liquidity + High-News Event Pool (Top 10)",
        "",
        f"- Window: {START_DATE} to {END_DATE}",
        f"- Theme: {args.theme}",
        f"- Min title-fit threshold: {args.min_title_fit}",
        "- Score: 0.5 * title_fit + 0.3 * normalized log-liquidity + 0.2 * normalized log-news-coverage",
        "- News matching: event-title token keywords against `Estimated_Title`",
        "",
        df.to_markdown(index=False),
        "",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved CSV: {out_csv}")
    print(f"Saved Report: {out_md}")
    print(df[["event_title", "title_fit_score", "liquidity_score_raw", "news_count", "combined_score"]].to_string(index=False))


if __name__ == "__main__":
    main()