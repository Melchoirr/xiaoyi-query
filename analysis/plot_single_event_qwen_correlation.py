import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


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


def market_token_id(market: dict) -> str:
    raw = market.get("clobTokenIds")
    if not raw:
        return ""
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
            return str(arr[0]) if arr else ""
        except Exception:
            return ""
    if isinstance(raw, list) and raw:
        return str(raw[0])
    return ""


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


def find_token_for_event(session: requests.Session, event_id: str, event_title: str, market_question: str) -> tuple[str, str]:
    payload = get_json(
        session,
        "https://gamma-api.polymarket.com/public-search",
        {"q": event_title, "limit_per_type": 50, "keep_closed_markets": 1},
    )
    events = payload.get("events", [])

    selected = None
    for ev in events:
        if str(ev.get("id", "")) == str(event_id):
            selected = ev
            break

    if selected is None and events:
        selected = events[0]

    if selected is None:
        return "", ""

    markets = selected.get("markets", []) or []
    if not markets:
        return "", ""

    best = None
    best_score = -1.0
    for m in markets:
        score = market_liquidity(m)
        q = str(m.get("question", ""))
        if market_question and q.strip().lower() == market_question.strip().lower():
            score += 1e9
        if score > best_score:
            best_score = score
            best = m

    token_id = market_token_id(best or {})
    chosen_question = str((best or {}).get("question", ""))
    return token_id, chosen_question


def fetch_price_history(session: requests.Session, token_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    if not token_id:
        return pd.DataFrame(columns=["datetime_utc", "price"])

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    cur = start_ts
    chunk_sec = 10 * 24 * 3600
    rows = []

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


def compute_lag_corr(vol: pd.Series, news: pd.Series, max_lag: int = 24) -> pd.DataFrame:
    data = []
    for lag in range(-max_lag, max_lag + 1):
        c = vol.corr(news.shift(lag))
        data.append({"lag": lag, "corr": float(c) if pd.notna(c) else np.nan})
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-event Qwen retrieval vs price-change correlation plots")
    parser.add_argument("--retrieval-csv", required=True)
    parser.add_argument("--max-lag", type=int, default=24)
    args = parser.parse_args()

    fp = Path(args.retrieval_csv)
    if not fp.exists():
        raise FileNotFoundError(f"retrieval csv not found: {fp}")

    df = pd.read_csv(fp)
    if df.empty:
        raise RuntimeError("retrieval csv is empty")

    event_id = str(df["event_id"].iloc[0])
    event_title = str(df["event_title"].iloc[0])
    market_question = str(df["selected_market_question"].iloc[0])

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    start_dt = df["datetime_utc"].min().floor("h")
    end_dt = df["datetime_utc"].max().ceil("h")
    grid = pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")

    agg_all = df.groupby(df["datetime_utc"].dt.floor("h"), as_index=False).agg(
        all_news_volume=("Estimated_Title", "count"),
        avg_sim=("sim", "mean"),
    )

    kept = df[df["kept_by_similarity"] == 1].copy()
    agg_keep = kept.groupby(kept["datetime_utc"].dt.floor("h"), as_index=False).agg(
        kept_news_volume=("Estimated_Title", "count"),
        kept_avg_sim=("sim", "mean"),
    )

    ts = pd.DataFrame({"datetime_utc": grid})
    ts = ts.merge(agg_all.rename(columns={"datetime_utc": "datetime_utc"}), on="datetime_utc", how="left")
    ts = ts.merge(agg_keep.rename(columns={"datetime_utc": "datetime_utc"}), on="datetime_utc", how="left")
    ts[["all_news_volume", "kept_news_volume"]] = ts[["all_news_volume", "kept_news_volume"]].fillna(0)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://polymarket.com",
            "Referer": "https://polymarket.com/",
        }
    )

    token_id, chosen_question = find_token_for_event(session, event_id, event_title, market_question)
    px_raw = fetch_price_history(session, token_id, start_dt, end_dt)
    price_available = not px_raw.empty

    if price_available:
        hourly_px = (
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

        ts = ts.merge(hourly_px, on="datetime_utc", how="left")
        p = ts["price"].clip(lower=1e-4, upper=1 - 1e-4)
        ts["logit_price"] = np.log(p / (1 - p))
        ts["ret_1h"] = ts["logit_price"].diff()
        ts["abs_ret_1h"] = ts["ret_1h"].abs()
        corr_keep = ts["abs_ret_1h"].corr(ts["kept_news_volume"])
        corr_all = ts["abs_ret_1h"].corr(ts["all_news_volume"])
        lag_df = compute_lag_corr(ts["abs_ret_1h"], ts["kept_news_volume"], max_lag=args.max_lag)
    else:
        ts["price"] = np.nan
        ts["logit_price"] = np.nan
        ts["ret_1h"] = np.nan
        ts["abs_ret_1h"] = np.nan
        corr_keep = np.nan
        corr_all = np.nan
        lag_df = pd.DataFrame({"lag": list(range(-args.max_lag, args.max_lag + 1)), "corr": np.nan})

    out_dir = Path("result") / "by_date" / start_dt.date().isoformat() / f"single_event_qwen_{event_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_out = out_dir / f"single_event_series_{event_id}_{start_dt.date().isoformat()}.csv"
    ts.to_csv(ts_out, index=False)

    sns.set_style("whitegrid")

    # 1) Time series overlay
    fig, ax1 = plt.subplots(figsize=(13, 5))
    if price_available:
        ax1.plot(ts["datetime_utc"], ts["price"], color="tab:blue", linewidth=1.8, label="Price")
        ax1.set_ylabel("Price", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title(f"Event {event_id} | Qwen Kept News vs Price")
        ax2 = ax1.twinx()
        ax2.bar(ts["datetime_utc"], ts["kept_news_volume"], color="tab:orange", alpha=0.35, width=0.03, label="Kept News Volume")
        ax2.set_ylabel("Kept News Volume", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
    else:
        ax1.bar(ts["datetime_utc"], ts["kept_news_volume"], color="tab:orange", alpha=0.55, width=0.03)
        ax1.set_ylabel("Kept News Volume")
        ax1.set_title(f"Event {event_id} | Qwen Kept News (Price unavailable)")

    fig.tight_layout()
    fig1 = out_dir / f"single_event_price_vs_kept_news_{event_id}.png"
    fig.savefig(fig1, dpi=220)
    plt.close(fig)

    # 2) Scatter with trend
    fig, ax = plt.subplots(figsize=(7, 5))
    if price_available:
        sns.regplot(
            x=ts["kept_news_volume"],
            y=ts["abs_ret_1h"],
            scatter_kws={"alpha": 0.4, "color": "tab:gray"},
            line_kws={"color": "tab:red", "linewidth": 2},
            ax=ax,
        )
        ax.set_title("Scatter: Kept News Volume vs abs_ret_1h")
        ax.set_xlabel("kept_news_volume (hourly)")
        ax.set_ylabel("abs_ret_1h")
        txt = f"Pearson corr={corr_keep:.4f}" if pd.notna(corr_keep) else "Pearson corr=NaN"
        ax.text(0.05, 0.92, txt, transform=ax.transAxes, fontsize=11, bbox=dict(facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "Price history unavailable\nScatter skipped", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    fig.tight_layout()
    fig2 = out_dir / f"single_event_scatter_corr_{event_id}.png"
    fig.savefig(fig2, dpi=220)
    plt.close(fig)

    # 3) Lag correlation
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if price_available:
        ax.bar(lag_df["lag"], lag_df["corr"], color="tab:purple", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("lag(hour)")
        ax.set_ylabel("corr(abs_ret_1h, kept_news_volume.shift(lag))")
        ax.set_title("Lead-Lag Correlation (Single Event)")
    else:
        ax.text(0.5, 0.5, "Price history unavailable\nLead-lag skipped", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    fig.tight_layout()
    fig3 = out_dir / f"single_event_lag_corr_{event_id}.png"
    fig.savefig(fig3, dpi=220)
    plt.close(fig)

    # 4) Similarity histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["sim"].dropna(), bins=50, alpha=0.75, color="tab:green")
    if "semantic_min_similarity" in df.columns:
        th = float(df["semantic_min_similarity"].iloc[0])
        ax.axvline(th, color="tab:red", linestyle="--", linewidth=1.8, label=f"threshold={th}")
        ax.legend()
    ax.set_xlabel("sim")
    ax.set_ylabel("count")
    ax.set_title("Qwen Similarity Distribution")
    fig.tight_layout()
    fig4 = out_dir / f"single_event_similarity_hist_{event_id}.png"
    fig.savefig(fig4, dpi=220)
    plt.close(fig)

    rep = out_dir / f"single_event_report_{event_id}.md"
    best_lag_row = lag_df.loc[lag_df["corr"].abs().idxmax()] if lag_df["corr"].notna().any() else None
    lines = [
        f"# Single Event Qwen Analysis ({event_id})",
        "",
        f"- event_title: {event_title}",
        f"- selected_market_question (retrieval csv): {market_question}",
        f"- selected_market_question (API chosen): {chosen_question}",
        f"- token_id: {token_id}",
        f"- retrieval_rows: {len(df)}",
        f"- kept_rows: {int((df['kept_by_similarity']==1).sum())}",
        f"- time_window_utc: {start_dt} -> {end_dt}",
        f"- price_available: {price_available}",
        f"- corr(abs_ret_1h, kept_news_volume): {corr_keep}",
        f"- corr(abs_ret_1h, all_news_volume): {corr_all}",
    ]

    if best_lag_row is not None and price_available:
        lines += [
            f"- best_lag_by_abs_corr: {int(best_lag_row['lag'])}",
            f"- best_lag_corr: {float(best_lag_row['corr'])}",
        ]

    lines += [
        "",
        "## Output Files",
        "",
        f"- {fig1}",
        f"- {fig2}",
        f"- {fig3}",
        f"- {fig4}",
        f"- {ts_out}",
        "",
        "## Notes",
        "",
        "- kept_rows very small will make correlation unstable; interpret as directional diagnostic, not causal evidence.",
    ]
    rep.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig1}")
    print(f"Saved: {fig2}")
    print(f"Saved: {fig3}")
    print(f"Saved: {fig4}")
    print(f"Saved: {ts_out}")
    print(f"Saved: {rep}")
    print(f"corr_keep={corr_keep}")
    print(f"corr_all={corr_all}")


if __name__ == "__main__":
    main()
