import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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


def to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def market_liquidity(market: dict) -> float:
    vals = [to_float(market.get(k)) for k in LIQUIDITY_KEYS]
    return max(vals) if vals else 0.0


def parse_token_ids(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
            return [str(x) for x in arr]
        except Exception:
            return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def get_json(session: requests.Session, url: str, params: dict, retries: int = 3) -> dict:
    last_err = None
    for _ in range(retries):
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception as err:
            last_err = err
        time.sleep(0.3)
    if last_err:
        raise RuntimeError(f"Request failed for {url}: {last_err}")
    return {}


def fetch_price_history(session: requests.Session, token_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    rows = []
    cur = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    chunk_sec = 10 * 24 * 3600

    while cur < end_ts:
        chunk_end = min(cur + chunk_sec, end_ts)
        payload = get_json(
            session,
            "https://clob.polymarket.com/prices-history",
            {"market": token_id, "startTs": cur, "endTs": chunk_end, "fidelity": 60},
        )
        rows.extend(payload.get("history", []))
        cur = chunk_end
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["datetime_utc", "price"])

    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["datetime_utc", "price"]).drop_duplicates("datetime_utc")
    return df[["datetime_utc", "price"]].sort_values("datetime_utc")


def find_working_market(session: requests.Session, event_id: str, event_title: str, preferred_question: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> tuple[str, str, pd.DataFrame]:
    payload = get_json(
        session,
        "https://gamma-api.polymarket.com/public-search",
        {"q": event_title, "limit_per_type": 50, "keep_closed_markets": 1},
    )
    events = payload.get("events", [])

    target_event = None
    for ev in events:
        if str(ev.get("id", "")) == str(event_id):
            target_event = ev
            break

    if target_event is None and events:
        target_event = events[0]

    if target_event is None:
        return "", "", pd.DataFrame(columns=["datetime_utc", "price"])

    markets = target_event.get("markets", []) or []
    if not markets:
        return "", "", pd.DataFrame(columns=["datetime_utc", "price"])

    def score(m):
        s = market_liquidity(m)
        q = str(m.get("question", "")).strip().lower()
        if preferred_question and q == preferred_question.strip().lower():
            s += 1e9
        return s

    markets = sorted(markets, key=score, reverse=True)

    for m in markets:
        q = str(m.get("question", ""))
        token_ids = parse_token_ids(m.get("clobTokenIds"))
        for token_id in token_ids:
            px = fetch_price_history(session, token_id, start_dt, end_dt)
            if not px.empty:
                return token_id, q, px

    return "", "", pd.DataFrame(columns=["datetime_utc", "price"])


def plot_like_small(df_merged: pd.DataFrame, keyword: str, out_png: Path) -> None:
    p_min = df_merged["price"].min()
    p_max = df_merged["price"].max()
    p_range = p_max - p_min
    margin = p_range * 0.1 if p_range > 0 else 0.05
    y_lower = max(0.0, p_min - margin)
    y_upper = min(1.0, p_max + margin)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.set_title(f"Alignment: Polymarket Price vs News Volume ({keyword})", fontsize=14)
    ax1.plot(df_merged["datetime_utc"], df_merged["price"], color="tab:blue", linewidth=2, label="Market Price", zorder=3)
    ax1.set_ylabel("Probability (Price)", color="tab:blue", fontsize=12, fontweight="bold")
    ax1.set_ylim(y_lower, y_upper)

    ax2 = ax1.twinx()
    ax2.bar(df_merged["datetime_utc"], df_merged["news_volume"], color="tab:gray", alpha=0.3, width=0.02, label="News Volume", zorder=1)
    ax2.set_ylabel("Hourly News Count", color="tab:gray")
    ax2.grid(False)

    ax3.set_title(f"Alignment: Polymarket Price vs News Sentiment ({keyword})", fontsize=14)
    ax3.plot(df_merged["datetime_utc"], df_merged["price"], color="tab:blue", linewidth=2, label="Market Price", zorder=3)
    ax3.set_ylabel("Probability (Price)", color="tab:blue", fontsize=12, fontweight="bold")
    ax3.set_ylim(y_lower, y_upper)

    ax4 = ax3.twinx()
    ax4.plot(df_merged["datetime_utc"], df_merged["avg_tone"], color="tab:orange", marker=".", linestyle="--", alpha=0.6, label="Sentiment (Tone)", zorder=2)
    ax4.axhline(0, color="red", linestyle=":", alpha=0.5)
    ax4.set_ylabel("Avg Sentiment Score", color="tab:orange")
    ax4.grid(False)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:00"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all tech events with polymarketAlignedGDELTSmall style")
    parser.add_argument("--event-pool-csv", default="result/by_date/2026-04-21/event_pool_top10_2026-04-21_tech.csv")
    parser.add_argument("--retrieval-dir", default="result/by_date/2026-04-21/news_retrieval_by_event_2026-04-21")
    parser.add_argument("--out-dir", default="result/by_date/2026-04-21/qwen_tech_style_plots")
    parser.add_argument("--start-date", default="2026-02-25 00:00:00")
    parser.add_argument("--end-date", default="2026-03-25 23:59:59")
    args = parser.parse_args()

    event_pool = pd.read_csv(args.event_pool_csv)
    retrieval_dir = Path(args.retrieval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = pd.to_datetime(args.start_date, utc=True)
    end_dt = pd.to_datetime(args.end_date, utc=True)
    grid = pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }
    )

    rows = []

    for _, ev in event_pool.iterrows():
        event_id = str(ev["event_id"])
        event_title = str(ev["event_title"])

        fp = retrieval_dir / f"event_{event_id}_retrieval_ranked_2026-04-21.csv"
        if not fp.exists():
            rows.append({"event_id": event_id, "event_title": event_title, "status": "missing_retrieval_csv", "png": ""})
            continue

        r = pd.read_csv(fp)
        if r.empty:
            rows.append({"event_id": event_id, "event_title": event_title, "status": "empty_retrieval_csv", "png": ""})
            continue

        r["datetime_utc"] = pd.to_datetime(r["datetime_utc"], utc=True).dt.floor("h")
        news = r.groupby("datetime_utc", as_index=False).agg(
            news_volume=("Estimated_Title", "count"),
            avg_tone=("tone", "mean"),
        )

        preferred_question = str(r["selected_market_question"].iloc[0]) if "selected_market_question" in r.columns else ""

        token_id, chosen_question, px_raw = find_working_market(
            session,
            event_id=event_id,
            event_title=event_title,
            preferred_question=preferred_question,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        if px_raw.empty:
            rows.append(
                {
                    "event_id": event_id,
                    "event_title": event_title,
                    "status": "no_price_history",
                    "token_id": token_id,
                    "selected_market_question": chosen_question,
                    "png": "",
                }
            )
            continue

        price = (
            px_raw.set_index("datetime_utc")["price"]
            .resample("h")
            .last()
            .reindex(grid)
            .ffill()
            .bfill()
            .rename("price")
            .reset_index()
            .rename(columns={"index": "datetime_utc"})
        )

        merged = pd.DataFrame({"datetime_utc": grid}).merge(price, on="datetime_utc", how="left")
        merged = merged.merge(news, on="datetime_utc", how="left")
        merged["news_volume"] = merged["news_volume"].fillna(0)
        merged["avg_tone"] = merged["avg_tone"].fillna(0)

        slug = re_sub_non_alnum(event_title)[:80]
        out_png = out_dir / f"event_{event_id}_{slug}_alignment_result.png"
        out_csv = out_dir / f"event_{event_id}_{slug}_merged_series.csv"

        plot_like_small(merged, event_title, out_png)
        merged.to_csv(out_csv, index=False)

        rows.append(
            {
                "event_id": event_id,
                "event_title": event_title,
                "status": "ok",
                "token_id": token_id,
                "selected_market_question": chosen_question,
                "png": str(out_png),
                "series_csv": str(out_csv),
                "news_total": int(news["news_volume"].sum()),
            }
        )
        logging.info(f"[Done] {event_id} -> {out_png}")

    summary = pd.DataFrame(rows)
    summary_csv = out_dir / "qwen_tech_style_plot_summary_2026-04-21.csv"
    summary_md = out_dir / "qwen_tech_style_plot_report_2026-04-21.md"
    summary.to_csv(summary_csv, index=False)

    lines = [
        "# Qwen Tech Event Plots (Small-script Style)",
        "",
        f"- Event pool: {args.event_pool_csv}",
        f"- Retrieval dir: {args.retrieval_dir}",
        f"- Success: {int((summary['status'] == 'ok').sum())}",
        f"- Failed: {int((summary['status'] != 'ok').sum())}",
        "",
        summary.to_markdown(index=False),
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_md}")


def re_sub_non_alnum(s: str) -> str:
    import re

    x = re.sub(r"[^A-Za-z0-9]+", "-", (s or "").lower())
    x = re.sub(r"-+", "-", x).strip("-")
    return x or "event"


if __name__ == "__main__":
    main()
