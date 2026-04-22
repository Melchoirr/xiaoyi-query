import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


START_DATE = "2026-02-25 00:00:00"
END_DATE = "2026-03-25 23:59:59"
DB_PATH = "data/gdelt_master.duckdb"

LIQUIDITY_KEYS = [
    "volumeNum",
    "volume",
    "volume24hr",
    "volume24hrClob",
    "oneDayVolume",
    "liquidityNum",
    "liquidity",
    "liquidityClob",
]

DEFAULT_SEMANTIC_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_CACHE: dict[str, object] = {}


@dataclass
class EventSummary:
    event_id: str
    event_title: str
    selected_market_question: str
    token_id: str
    raw_trade_points: int
    observed_hours: int
    total_hours: int
    observed_ratio: float
    news_hours: int
    news_total_articles: int
    lexical_overlap_score: float
    low_match_flag: int


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def title_tokens(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9']+", (text or "").lower())
    return {t for t in toks if len(t) >= 4}


def lexical_overlap(event_title: str, market_question: str) -> float:
    lhs = title_tokens(event_title)
    if not lhs:
        return 0.0
    rhs = title_tokens(market_question)
    return len(lhs.intersection(rhs)) / len(lhs)


def get_json(session: requests.Session, url: str, params: dict, retries: int = 3) -> dict:
    last_err = None
    for _ in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.json()
        except Exception as err:
            last_err = err
        time.sleep(0.3)
    if last_err:
        raise RuntimeError(f"Request failed for {url} with params={params}: {last_err}")
    return {}


def find_event_markets(session: requests.Session, event_id: str, event_title: str) -> list[dict]:
    payload = get_json(
        session,
        "https://gamma-api.polymarket.com/public-search",
        {"q": event_title, "limit_per_type": 30, "keep_closed_markets": 1},
    )
    events = payload.get("events", [])

    # First try strict id match, fallback to best title match.
    for ev in events:
        if str(ev.get("id", "")) == str(event_id):
            return ev.get("markets", []) or []

    title_l = event_title.strip().lower()
    for ev in events:
        if (ev.get("title", "").strip().lower() == title_l):
            return ev.get("markets", []) or []

    return []


def market_liquidity(market: dict) -> float:
    vals = [to_float(market.get(k)) for k in LIQUIDITY_KEYS]
    return max(vals) if vals else 0.0


def market_token_id(market: dict) -> str:
    raw = market.get("clobTokenIds")
    if not raw:
        return ""
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
            if arr:
                return str(arr[0])
        except Exception:
            return ""
    if isinstance(raw, list) and raw:
        return str(raw[0])
    return ""


def fetch_price_history(
    session: requests.Session,
    token_id: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    if not token_id:
        return pd.DataFrame(columns=["datetime_utc", "price"]) 

    all_hist = []
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    cur = start_ts
    chunk_sec = 10 * 24 * 3600

    while cur < end_ts:
        chunk_end = min(cur + chunk_sec, end_ts)
        payload = get_json(
            session,
            "https://clob.polymarket.com/prices-history",
            {
                "market": token_id,
                "startTs": cur,
                "endTs": chunk_end,
                "fidelity": 60,
            },
        )
        all_hist.extend(payload.get("history", []))
        cur = chunk_end
        time.sleep(0.15)

    if not all_hist:
        return pd.DataFrame(columns=["datetime_utc", "price"])

    df = pd.DataFrame(all_hist)
    df["datetime_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["datetime_utc", "price"]).drop_duplicates("datetime_utc")
    return df[["datetime_utc", "price"]].sort_values("datetime_utc")


def build_price_series(raw_df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    grid = pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")
    out = pd.DataFrame({"datetime_utc": grid})
    if raw_df.empty:
        out["price"] = np.nan
        out["observed_trade_hour"] = 0
        return out, 0

    hourly_obs = (
        raw_df.set_index("datetime_utc")["price"]
        .resample("h")
        .last()
    )
    observed = hourly_obs.reindex(grid)
    out["observed_trade_hour"] = observed.notna().astype(int).values
    out["price"] = observed.ffill().bfill().values
    return out, int(len(raw_df))


def fetch_news_features(
    con: duckdb.DuckDBPyConnection,
    keywords_csv: str,
    event_query: str,
    start_str: str,
    end_str: str,
    retrieval_mode: str = "hybrid",
    semantic_min_similarity: float = 0.6,
    semantic_model_name: str = DEFAULT_SEMANTIC_MODEL,
    max_candidates: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked_cols = ["datetime_utc", "Estimated_Title", "tone", "kw_score", "sim", "kept_by_similarity"]
    keywords = [k.strip() for k in keywords_csv.split(",") if k.strip()]
    if not keywords:
        return (
            pd.DataFrame(columns=["datetime_utc", "news_volume", "avg_tone"]),
            pd.DataFrame(columns=ranked_cols),
        )

    start_key = start_str.replace("-", "").replace(" ", "").replace(":", "")
    end_key = end_str.replace("-", "").replace(" ", "").replace(":", "")

    like_clauses = []
    for kw in keywords:
        safe = kw.replace("'", "''")
        like_clauses.append(f"Estimated_Title ILIKE '%{safe}%'")
    where_keywords = " OR ".join(like_clauses)

    kw_score_parts = []
    for kw in keywords:
        safe = kw.replace("'", "''")
        kw_score_parts.append(f"CASE WHEN Estimated_Title ILIKE '%{safe}%' THEN 1 ELSE 0 END")
    kw_score_expr = " + ".join(kw_score_parts) if kw_score_parts else "0"

    limit_clause = f"LIMIT {int(max_candidates)}" if int(max_candidates) > 0 else ""
    sql = f"""
        SELECT
            strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc,
            Estimated_Title,
            CAST(split_part(V2Tone, ',', 1) AS FLOAT) AS tone,
            ({kw_score_expr}) AS kw_score
        FROM gkg_news
        WHERE CAST(DATE AS VARCHAR) >= '{start_key}'
          AND CAST(DATE AS VARCHAR) <= '{end_key}'
          AND ({where_keywords})
        {limit_clause}
    """
    df_raw = con.execute(sql).df()
    if df_raw.empty:
        return (
            pd.DataFrame(columns=["datetime_utc", "news_volume", "avg_tone"]),
            pd.DataFrame(columns=ranked_cols),
        )

    df_raw["datetime_utc"] = pd.to_datetime(df_raw["datetime_utc"]).dt.tz_localize("UTC")
    df_raw["Estimated_Title"] = df_raw["Estimated_Title"].fillna("")

    # Layer-1 only: lexical aggregation baseline.
    if retrieval_mode == "keyword":
        ranked = df_raw.copy()
        ranked["sim"] = np.nan
        ranked["kept_by_similarity"] = 1
        ranked = ranked.sort_values(["kw_score", "datetime_utc"], ascending=[False, True])
        agg = (
            df_raw.groupby("datetime_utc", as_index=False)
            .agg(news_volume=("Estimated_Title", "count"), avg_tone=("tone", "mean"))
            .sort_values("datetime_utc")
        )
        return agg, ranked[ranked_cols]

    # Layer-2 semantic rerank.
    titles = df_raw["Estimated_Title"].tolist()
    query = (event_query or "").strip()
    if not query:
        query = keywords_csv.replace(",", " ")

    sim = None
    if SentenceTransformer is not None:
        try:
            model = _MODEL_CACHE.get(semantic_model_name)
            if model is None:
                model = SentenceTransformer(semantic_model_name)
                _MODEL_CACHE[semantic_model_name] = model
            q_emb = model.encode([query], normalize_embeddings=True)
            t_emb = model.encode(titles, normalize_embeddings=True)
            sim = (t_emb @ q_emb[0]).astype(float)
        except Exception:
            sim = None

    # Fallback vector rerank if sentence-transformer is unavailable.
    if sim is None:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        mat = vec.fit_transform([query] + titles)
        sim = cosine_similarity(mat[0:1], mat[1:]).reshape(-1)

    df_raw["sim"] = sim
    threshold = float(max(-1.0, min(1.0, semantic_min_similarity)))
    ranked = df_raw.copy()
    ranked["kept_by_similarity"] = (ranked["sim"] >= threshold).astype(int)
    ranked = ranked.sort_values(["sim", "kw_score", "datetime_utc"], ascending=[False, False, True])
    df_keep = df_raw[df_raw["sim"] >= threshold].copy()
    if df_keep.empty:
        return (
            pd.DataFrame(columns=["datetime_utc", "news_volume", "avg_tone"]),
            ranked[ranked_cols],
        )

    agg = (
        df_keep.groupby("datetime_utc", as_index=False)
        .agg(news_volume=("Estimated_Title", "count"), avg_tone=("tone", "mean"))
        .sort_values("datetime_utc")
    )
    return agg, ranked[ranked_cols]


def add_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["price"] = pd.to_numeric(x["price"], errors="coerce")

    # Stable logit transform for probability prices.
    p = x["price"].clip(lower=1e-4, upper=1 - 1e-4)
    x["logit_price"] = np.log(p / (1 - p))
    x["ret_1h"] = x["logit_price"].diff()
    x["abs_ret_1h"] = x["ret_1h"].abs()

    x["news_volume"] = x["news_volume"].fillna(0)
    x["avg_tone"] = x["avg_tone"].fillna(0)
    x["news_ma24"] = x["news_volume"].rolling(window=24, min_periods=1).mean()
    x["news_std24"] = x["news_volume"].rolling(window=24, min_periods=1).std().replace(0, 1).fillna(1)
    x["news_zscore"] = (x["news_volume"] - x["news_ma24"]) / x["news_std24"]
    return x


def select_market_and_build(
    session: requests.Session,
    con: duckdb.DuckDBPyConnection,
    event_id: str,
    event_title: str,
    news_keywords: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    retrieval_mode: str,
    semantic_min_similarity: float,
    semantic_model_name: str,
    max_candidates: int,
) -> tuple[pd.DataFrame, EventSummary, pd.DataFrame]:
    markets = find_event_markets(session, event_id=event_id, event_title=event_title)
    if not markets:
        empty_grid = pd.DataFrame({"datetime_utc": pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")})
        empty_grid["price"] = np.nan
        empty_grid["observed_trade_hour"] = 0
        event_query = f"{event_title}".strip()
        news, ranked = fetch_news_features(
            con,
            news_keywords,
            event_query,
            START_DATE,
            END_DATE,
            retrieval_mode=retrieval_mode,
            semantic_min_similarity=semantic_min_similarity,
            semantic_model_name=semantic_model_name,
            max_candidates=max_candidates,
        )
        ranked["event_id"] = event_id
        ranked["event_title"] = event_title
        ranked["selected_market_question"] = ""
        ranked["semantic_min_similarity"] = semantic_min_similarity
        merged = empty_grid.merge(news, on="datetime_utc", how="left")
        merged = add_ts_features(merged)
        merged["event_id"] = event_id
        merged["event_title"] = event_title
        merged["selected_market_question"] = ""
        merged["token_id"] = ""
        summary = EventSummary(
            event_id=event_id,
            event_title=event_title,
            selected_market_question="",
            token_id="",
            raw_trade_points=0,
            observed_hours=0,
            total_hours=len(merged),
            observed_ratio=0.0,
            news_hours=int((merged["news_volume"].fillna(0) > 0).sum()),
            news_total_articles=int(merged["news_volume"].fillna(0).sum()),
            lexical_overlap_score=0.0,
            low_match_flag=1,
        )
        return merged, summary, ranked

    # Keep cost manageable on very large events.
    markets_sorted = sorted(markets, key=market_liquidity, reverse=True)[:5]
    best = None
    best_df = None
    best_raw_points = -1
    best_observed_hours = -1
    best_overlap = -1.0

    for m in markets_sorted:
        token_id = market_token_id(m)
        raw = fetch_price_history(session, token_id=token_id, start_dt=start_dt, end_dt=end_dt)
        series_df, raw_points = build_price_series(raw, start_dt=start_dt, end_dt=end_dt)
        observed_hours = int(series_df["observed_trade_hour"].sum())
        overlap = lexical_overlap(event_title, m.get("question", ""))
        if (
            observed_hours > best_observed_hours
            or (observed_hours == best_observed_hours and overlap > best_overlap)
            or (observed_hours == best_observed_hours and math.isclose(overlap, best_overlap) and raw_points > best_raw_points)
        ):
            best = m
            best_df = series_df
            best_raw_points = raw_points
            best_observed_hours = observed_hours
            best_overlap = overlap

    if best is None or best_df is None:
        raise RuntimeError(f"No market could be selected for event_id={event_id}")

    event_query = f"{event_title} {best.get('question', '')}".strip()
    news, ranked = fetch_news_features(
        con,
        news_keywords,
        event_query,
        START_DATE,
        END_DATE,
        retrieval_mode=retrieval_mode,
        semantic_min_similarity=semantic_min_similarity,
        semantic_model_name=semantic_model_name,
        max_candidates=max_candidates,
    )
    ranked["event_id"] = event_id
    ranked["event_title"] = event_title
    ranked["selected_market_question"] = best.get("question", "")
    ranked["semantic_min_similarity"] = semantic_min_similarity
    merged = best_df.merge(news, on="datetime_utc", how="left")
    merged = add_ts_features(merged)
    merged["event_id"] = event_id
    merged["event_title"] = event_title
    merged["selected_market_question"] = best.get("question", "")
    merged["token_id"] = market_token_id(best)

    summary = EventSummary(
        event_id=event_id,
        event_title=event_title,
        selected_market_question=best.get("question", ""),
        token_id=market_token_id(best),
        raw_trade_points=int(best_raw_points),
        observed_hours=int(best_observed_hours),
        total_hours=int(len(merged)),
        observed_ratio=float(best_observed_hours / len(merged)) if len(merged) else 0.0,
        news_hours=int((merged["news_volume"].fillna(0) > 0).sum()),
        news_total_articles=int(merged["news_volume"].fillna(0).sum()),
        lexical_overlap_score=float(best_overlap),
        low_match_flag=int(best_overlap < 0.35),
    )
    return merged, summary, ranked


def write_markdown_summary(summary_df: pd.DataFrame, md_path: str, panel_path: str) -> None:
    lines = [
        "# Unified Top10 Event Panel Summary",
        "",
        f"- Window: {START_DATE} to {END_DATE}",
        "- Grid: hourly UTC",
        "- Price features: logit_price, ret_1h, abs_ret_1h",
        "- News features: news_volume, avg_tone, news_zscore",
        f"- Panel file: {panel_path}",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified panel dataset for top-10 event pool")
    parser.add_argument(
        "--event-pool-csv",
        default=f"result/event_pool_top10_{date.today().isoformat()}.csv",
        help="Path to top-10 event pool csv",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["keyword", "hybrid"],
        default="hybrid",
        help="News retrieval mode: keyword only or layered hybrid retrieval",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional candidate cap before semantic rerank; 0 means no truncation",
    )
    parser.add_argument(
        "--semantic-min-similarity",
        type=float,
        default=0.6,
        help="Minimum cosine similarity for hybrid semantic filter",
    )
    parser.add_argument(
        "--semantic-model-name",
        type=str,
        default=DEFAULT_SEMANTIC_MODEL,
        help="Sentence-Transformers model name for semantic retrieval",
    )
    args = parser.parse_args()

    events = pd.read_csv(args.event_pool_csv)
    start_dt = pd.to_datetime(START_DATE, utc=True)
    end_dt = pd.to_datetime(END_DATE, utc=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }
    )

    con = duckdb.connect(DB_PATH, read_only=True)

    panel_parts = []
    summary_rows = []
    ranked_news_parts = []
    for row in events.itertuples(index=False):
        panel_df, summary, ranked_df = select_market_and_build(
            session=session,
            con=con,
            event_id=str(row.event_id),
            event_title=str(row.event_title),
            news_keywords=str(row.news_keywords),
            start_dt=start_dt,
            end_dt=end_dt,
            retrieval_mode=args.retrieval_mode,
            semantic_min_similarity=args.semantic_min_similarity,
            semantic_model_name=args.semantic_model_name,
            max_candidates=args.max_candidates,
        )
        panel_parts.append(panel_df)
        summary_rows.append(summary.__dict__)
        ranked_news_parts.append(ranked_df)

    con.close()

    panel = pd.concat(panel_parts, ignore_index=True)
    panel = panel[
        [
            "datetime_utc",
            "event_id",
            "event_title",
            "selected_market_question",
            "token_id",
            "price",
            "observed_trade_hour",
            "logit_price",
            "ret_1h",
            "abs_ret_1h",
            "news_volume",
            "avg_tone",
            "news_zscore",
        ]
    ]

    today = date.today().isoformat()
    out_dir = Path("result") / "by_date" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / f"event_panel_top10_{today}.csv"
    summary_path = out_dir / f"event_panel_summary_top10_{today}.csv"
    summary_md_path = out_dir / f"event_panel_summary_top10_{today}.md"
    ranked_news_path = out_dir / f"news_retrieval_ranked_top10_{today}.csv"
    ranked_preview_path = out_dir / f"news_retrieval_ranked_preview_top10_{today}.csv"
    ranked_by_event_dir = out_dir / f"news_retrieval_by_event_{today}"
    ranked_by_event_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(panel_path, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    write_markdown_summary(summary_df, str(summary_md_path), str(panel_path))

    ranked_news = pd.concat(ranked_news_parts, ignore_index=True)
    ranked_news = ranked_news[
        [
            "event_id",
            "event_title",
            "selected_market_question",
            "datetime_utc",
            "Estimated_Title",
            "tone",
            "kw_score",
            "sim",
            "kept_by_similarity",
            "semantic_min_similarity",
        ]
    ]
    ranked_news = ranked_news.sort_values(
        ["sim", "kept_by_similarity", "kw_score", "datetime_utc"],
        ascending=[False, False, False, True],
    )
    ranked_news.to_csv(ranked_news_path, index=False)

    # Compact preview to inspect retrieval quality quickly.
    ranked_preview = (
        ranked_news.groupby("event_id", as_index=False, group_keys=False)
        .head(200)
        .reset_index(drop=True)
    )
    ranked_preview.to_csv(ranked_preview_path, index=False)

    # Per-event files make manual inspection easier than one giant csv.
    for event_id, g in ranked_news.groupby("event_id", sort=True):
        out_event_file = ranked_by_event_dir / f"event_{str(event_id)}_retrieval_ranked_{today}.csv"
        g.to_csv(out_event_file, index=False)

    print(f"Saved panel: {panel_path}")
    print(f"Saved summary csv: {summary_path}")
    print(f"Saved summary md: {summary_md_path}")
    print(f"Saved ranked retrieval csv: {ranked_news_path}")
    print(f"Saved ranked retrieval preview: {ranked_preview_path}")
    print(f"Saved ranked retrieval by-event dir: {ranked_by_event_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()