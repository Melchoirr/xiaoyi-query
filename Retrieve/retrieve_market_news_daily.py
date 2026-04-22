import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


DEFAULT_DB_PATH = "data/gdelt_master.duckdb"
DEFAULT_OUTPUT_DIR = "Retrieve"
STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "will", "would", "could", "should", "by", "with", "from", "at", "as", "about", "into", "after",
    "before", "between", "than", "that", "this", "these", "those", "it", "its", "be", "been", "being",
}


def default_date_window_utc() -> tuple[str, str]:
    return "2026-02-25 00:00:00", "2026-03-25 23:59:59"


@dataclass
class RetrievalConfig:
    search_query: str
    start_date: str
    end_date: str
    db_path: str = DEFAULT_DB_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    keyword_hints: list[str] | None = None
    top_k_day: int = 100
    max_candidates: int = 200000
    lexical_topn: int = 8000
    embedding_batch_size: int = 128
    model_name: str = "BAAI/bge-large-en-v1.5"
    hybrid_alpha: float = 0.75
    min_embedding_sim: float = 0.45
    fallback_keep_ratio: float = 0.08


def prompt_text(label: str, default: str = "", required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        if not required:
            return ""
        print("Input required. Please provide a value.")


def prompt_int(label: str, default: int) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter a valid integer.")


def prompt_float(label: str, default: float) -> float:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("Please enter a valid number.")


def parse_interactive_config() -> RetrievalConfig:
    print("=== Embedding News Retrieval (Interactive) ===")
    default_start_date, default_end_date = default_date_window_utc()
    search_query = prompt_text("search_query (news topic statement)", required=True)
    start_date = prompt_text("start_date (UTC, e.g. 2026-02-25 00:00:00)", default_start_date)
    end_date = prompt_text("end_date (UTC, e.g. 2026-03-25 23:59:59)", default_end_date)
    db_path = prompt_text("db_path", DEFAULT_DB_PATH)
    output_dir = prompt_text("output_dir", DEFAULT_OUTPUT_DIR)
    hints_text = prompt_text("keyword_hints (comma separated)", "")
    keyword_hints = [x.strip() for x in hints_text.split(",") if x.strip()]

    top_k_day = prompt_int("top_k_day", 100)
    max_candidates = prompt_int("max_candidates (SQL coarse recall cap, 0 means no limit)", 200000)
    lexical_topn = prompt_int("lexical_topn (candidates kept before embedding)", 8000)
    embedding_batch_size = prompt_int("embedding_batch_size", 128)
    model_name = prompt_text("model_name", "BAAI/bge-large-en-v1.5")
    hybrid_alpha = prompt_float("hybrid_alpha", 0.75)
    min_embedding_sim = prompt_float("min_embedding_sim", 0.45)
    fallback_keep_ratio = prompt_float("fallback_keep_ratio", 0.08)

    return RetrievalConfig(
        search_query=search_query,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        output_dir=output_dir,
        keyword_hints=keyword_hints,
        top_k_day=top_k_day,
        max_candidates=max_candidates,
        lexical_topn=lexical_topn,
        embedding_batch_size=embedding_batch_size,
        model_name=model_name,
        hybrid_alpha=hybrid_alpha,
        min_embedding_sim=min_embedding_sim,
        fallback_keep_ratio=fallback_keep_ratio,
    )


def to_gdelt_key(ts: str) -> str:
    return ts.replace("-", "").replace(" ", "").replace(":", "")


def safe_slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower())
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "query"
    return s[:max_len]


def extract_title_from_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        path = urlparse(url).path
        slug = path.strip("/").split("/")[-1]
        slug = re.sub(r"\.(html|htm|php|asp|aspx|cms|amp|stml)$", "", slug, flags=re.IGNORECASE)
        clean = re.sub(r"[-_]+", " ", slug).strip()
        if len(clean) < 6:
            return ""
        return clean
    except Exception:
        return ""


def build_recall_terms(search_query: str, keyword_hints: list[str] | None) -> list[str]:
    if keyword_hints:
        return [x.strip().lower() for x in keyword_hints if x.strip()]

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']+", (search_query or "").lower())
    terms = []
    seen = set()
    for tok in tokens:
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return terms


def query_candidates(
    con: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
    search_query: str,
    keyword_hints: list[str],
    max_candidates: int,
) -> pd.DataFrame:
    start_key = to_gdelt_key(start_date)
    end_key = to_gdelt_key(end_date)

    where_parts = [
        f"CAST(DATE AS VARCHAR) >= '{start_key}'",
        f"CAST(DATE AS VARCHAR) <= '{end_key}'",
        "DocumentIdentifier IS NOT NULL",
    ]

    recall_terms = build_recall_terms(search_query, keyword_hints)
    if recall_terms:
        likes = []
        for kw in recall_terms:
            esc = kw.replace("'", "''")
            likes.append(f"Estimated_Title ILIKE '%{esc}%' OR DocumentIdentifier ILIKE '%{esc}%'")
        where_parts.append("(" + " OR ".join(likes) + ")")

    order_clause = "ORDER BY CAST(DATE AS VARCHAR) DESC"
    limit_clause = f"LIMIT {int(max_candidates)}" if max_candidates > 0 else ""
    sql = f"""
        SELECT
            strptime(SUBSTR(CAST(DATE AS VARCHAR), 1, 10), '%Y%m%d%H') AS datetime_utc,
            SourceCommonName AS source,
            DocumentIdentifier AS url,
            Estimated_Title AS title,
            CAST(split_part(V2Tone, ',', 1) AS FLOAT) AS tone
        FROM gkg_news
        WHERE {' AND '.join(where_parts)}
        {order_clause}
        {limit_clause}
    """

    df = con.execute(sql).df()
    if df.empty:
        return df

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize("UTC")
    df["title"] = df["title"].fillna("")
    url_title = df["url"].fillna("").map(extract_title_from_url)
    df.loc[df["title"].str.len() < 6, "title"] = url_title[df["title"].str.len() < 6]
    df = df[df["title"].fillna("").str.len() >= 6].copy()
    print(f"[Stage-1 SQL coarse recall] terms={len(recall_terms)} candidates={len(df)}")
    return df


def rank_candidates(
    df: pd.DataFrame,
    search_query: str,
    model_name: str,
    hybrid_alpha: float,
    min_embedding_sim: float,
    fallback_keep_ratio: float,
    lexical_topn: int,
    embedding_batch_size: int,
) -> tuple[pd.DataFrame, int]:
    query = search_query.strip()
    if not query:
        raise ValueError("search_query cannot be empty")

    titles = df["title"].tolist()

    # Stage-2 lexical prefilter.
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    mat = vec.fit_transform([query] + titles)
    lex_scores = cosine_similarity(mat[0:1], mat[1:]).reshape(-1)

    lexical_keep_n = int(max(100, lexical_topn))
    if len(df) > lexical_keep_n:
        lex_idx = pd.Series(lex_scores).nlargest(lexical_keep_n).index
        work = df.iloc[lex_idx].copy().reset_index(drop=True)
        work_lex = lex_scores[lex_idx]
    else:
        work = df.copy().reset_index(drop=True)
        work_lex = lex_scores

    print(f"[Stage-2 lexical prefilter] input={len(df)} kept={len(work)}")

    # Stage-3 embedding rerank on reduced set.
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed in current environment. "
            "Please install: pip install -U torch sentence-transformers transformers"
        )

    model_name_l = model_name.lower()
    query_for_embed = query
    docs_for_embed = work["title"].tolist()

    # Model-aware input formatting improves retrieval quality for instruction-tuned models.
    if "bge" in model_name_l:
        query_for_embed = f"Represent this sentence for searching relevant passages: {query}"
    elif "e5" in model_name_l:
        query_for_embed = f"query: {query}"
        docs_for_embed = [f"passage: {x}" for x in docs_for_embed]

    try:
        model = SentenceTransformer(model_name)
        q_emb = model.encode([query_for_embed], normalize_embeddings=True)
        t_emb = model.encode(
            docs_for_embed,
            normalize_embeddings=True,
            batch_size=max(8, int(embedding_batch_size)),
            show_progress_bar=True,
        )
        emb_scores = (t_emb @ q_emb[0]).astype(float)
    except Exception as err:
        raise RuntimeError(f"Embedding encode failed: {err}") from err

    alpha = float(max(0.0, min(1.0, hybrid_alpha)))
    out = work.copy()
    out["embedding_sim"] = emb_scores
    out["lexical_sim"] = work_lex
    out["hybrid_score"] = alpha * out["embedding_sim"] + (1 - alpha) * out["lexical_sim"]

    keep = out[out["embedding_sim"] >= float(min_embedding_sim)].copy()
    if keep.empty:
        ratio = float(max(0.01, min(0.5, fallback_keep_ratio)))
        keep_n = max(1, int(len(out) * ratio))
        keep = out.nlargest(keep_n, "hybrid_score").copy()

    print(f"[Stage-3 embedding rerank] kept={len(keep)}")
    return keep, len(work)


def select_daily_topk(df_ranked: pd.DataFrame, top_k_day: int) -> pd.DataFrame:
    out = df_ranked.copy()
    out["date_utc"] = out["datetime_utc"].dt.date
    out = out.sort_values(["date_utc", "hybrid_score"], ascending=[True, False])
    out["rank_in_day"] = out.groupby("date_utc").cumcount() + 1
    out = out[out["rank_in_day"] <= int(max(1, top_k_day))].copy()
    out["relevance_label"] = ""
    out["review_notes"] = ""
    cols = [
        "date_utc",
        "rank_in_day",
        "datetime_utc",
        "hybrid_score",
        "embedding_sim",
        "lexical_sim",
        "tone",
        "source",
        "title",
        "url",
        "relevance_label",
        "review_notes",
    ]
    return out[cols].sort_values(["date_utc", "rank_in_day"])


def prepare_ranked_output(df_ranked: pd.DataFrame) -> pd.DataFrame:
    out = df_ranked.copy().sort_values("hybrid_score", ascending=False)
    out["date_utc"] = out["datetime_utc"].dt.date
    out["global_rank"] = range(1, len(out) + 1)
    out["relevance_label"] = ""
    out["review_notes"] = ""
    cols = [
        "global_rank",
        "date_utc",
        "datetime_utc",
        "hybrid_score",
        "embedding_sim",
        "lexical_sim",
        "tone",
        "source",
        "title",
        "url",
        "relevance_label",
        "review_notes",
    ]
    return out[cols]


def daily_summary(df_daily: pd.DataFrame) -> pd.DataFrame:
    return (
        df_daily.groupby("date_utc", as_index=False)
        .agg(
            kept_docs=("title", "count"),
            mean_hybrid_score=("hybrid_score", "mean"),
            mean_embedding_sim=("embedding_sim", "mean"),
            mean_lexical_sim=("lexical_sim", "mean"),
            unique_sources=("source", "nunique"),
            mean_tone=("tone", "mean"),
        )
        .sort_values("date_utc")
    )


def build_report_text(
    cfg: RetrievalConfig,
    candidate_n: int,
    lexical_prefilter_n: int,
    ranked_n: int,
    daily_topk_n: int,
    summary_df: pd.DataFrame,
) -> str:
    coverage_days = int(summary_df["date_utc"].nunique()) if not summary_df.empty else 0
    lines = [
        "# Embedding Retrieval Report",
        "",
        "## Retrieval Setup",
        "",
        f"- search_query: {cfg.search_query}",
        f"- date_window_utc: {cfg.start_date} to {cfg.end_date}",
        f"- db_path: {cfg.db_path}",
        f"- model_name: {cfg.model_name}",
        f"- hybrid_alpha: {cfg.hybrid_alpha}",
        f"- min_embedding_sim: {cfg.min_embedding_sim}",
        f"- top_k_day: {cfg.top_k_day}",
        f"- max_candidates: {cfg.max_candidates}",
        f"- lexical_topn: {cfg.lexical_topn}",
        f"- embedding_batch_size: {cfg.embedding_batch_size}",
        "",
        "## Retrieval Outcome",
        "",
        f"- candidates_from_duckdb: {candidate_n}",
        f"- lexical_prefilter_kept: {lexical_prefilter_n}",
        f"- semantic_kept_global: {ranked_n}",
        f"- final_daily_rows: {daily_topk_n}",
        f"- coverage_days_with_hits: {coverage_days}",
        "",
        "## Daily Statistics",
        "",
    ]

    if summary_df.empty:
        lines.append("No daily records passed filters.")
    else:
        lines.append(summary_df.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Accuracy Study Guidance",
            "",
            "- Use relevance_label with 1/0 for manual relevance judgments.",
            "- Compute Precision@K by day and macro-average across days.",
            "- review_notes can record false-positive reasons.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    cfg = parse_interactive_config()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(cfg.db_path, read_only=True)
    df_candidates = query_candidates(
        con=con,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        search_query=cfg.search_query,
        keyword_hints=cfg.keyword_hints or [],
        max_candidates=cfg.max_candidates,
    )
    con.close()

    if df_candidates.empty:
        raise RuntimeError("No candidate articles found in the selected time window/filter.")

    df_ranked, lexical_prefilter_n = rank_candidates(
        df=df_candidates,
        search_query=cfg.search_query,
        model_name=cfg.model_name,
        hybrid_alpha=cfg.hybrid_alpha,
        min_embedding_sim=cfg.min_embedding_sim,
        fallback_keep_ratio=cfg.fallback_keep_ratio,
        lexical_topn=cfg.lexical_topn,
        embedding_batch_size=cfg.embedding_batch_size,
    )

    df_daily = select_daily_topk(df_ranked, top_k_day=cfg.top_k_day)
    df_summary = daily_summary(df_daily)

    today = date.today().isoformat()
    slug = safe_slug(cfg.search_query)
    ranked_csv = out_dir / f"embedding_ranked_news_{slug}_{today}.csv"
    detail_csv = out_dir / f"embedding_daily_news_{slug}_{today}.csv"
    summary_csv = out_dir / f"embedding_daily_summary_{slug}_{today}.csv"
    report_md = out_dir / f"embedding_retrieval_report_{slug}_{today}.md"

    prepare_ranked_output(df_ranked).to_csv(ranked_csv, index=False)
    df_daily.to_csv(detail_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    report = build_report_text(
        cfg=cfg,
        candidate_n=len(df_candidates),
        lexical_prefilter_n=lexical_prefilter_n,
        ranked_n=len(df_ranked),
        daily_topk_n=len(df_daily),
        summary_df=df_summary,
    )
    report_md.write_text(report, encoding="utf-8")

    print(f"Saved global ranked retrieval: {ranked_csv}")
    print(f"Saved daily retrieval: {detail_csv}")
    print(f"Saved daily summary: {summary_csv}")
    print(f"Saved report: {report_md}")


if __name__ == "__main__":
    main()
