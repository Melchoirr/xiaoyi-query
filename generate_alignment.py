"""Generate merged price-news series and alignment plots for all events.

Uses:
- Knowledge graph (entities.json/knowledge_graph.json) for entity-based GDELT matching
- Existing merged series CSVs for Polymarket price data
- DuckDB gdelt_master for GDELT news matching

Output:
- {event_id}_{title}_merged_series.csv  (datetime_utc, price, news_volume, avg_tone)
- {event_id}_{title}_alignment_result.png  (dual-axis plot)
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gdelt_matcher import RealGDELTMatcher, load_graph_from_file
from config import DATA_DIR

# Path to existing merged CSVs (for price data extraction)
EXISTING_RESULT_DIR = Path(__file__).parent / "result/by_date/2026-04-23/qwen_initial_political_events_style_plots"


def extract_price_from_existing_csv(event_id):
    """Extract price data from existing merged series CSV."""
    pattern = f"event_{event_id}_"
    for f in EXISTING_RESULT_DIR.glob(f"{pattern}*_merged_series.csv"):
        df = pd.read_csv(f)
        if "datetime_utc" in df.columns and "price" in df.columns:
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
            return df[["datetime_utc", "price"]].copy()
    return None


def generate_merged_series(graph, matcher, output_dir):
    """For each event in the knowledge graph, merge price + GDELT news."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for event_id in sorted(graph["event_entity_index"].keys()):
        # Get event title
        event_title = event_id
        for evt in graph.get("_events_raw", []):
            if evt.get("event_id") == event_id:
                event_title = evt.get("title", event_id)
                break

        safe_name = event_title.lower().replace(" ", "-")[:80]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "" for c in safe_name)

        print(f"\n  Processing: {event_title[:70]}")

        # Get price data from existing CSV
        price_df = extract_price_from_existing_csv(event_id)

        # Get GDELT news data
        result = matcher.match_news(event_id)
        articles = result["articles"]
        search_terms = result["search_terms"]
        print(f"    Search terms: {search_terms}")
        print(f"    Articles matched: {len(articles)}")

        if not articles:
            print(f"    WARNING: No articles found for {event_id}")
            if price_df is not None:
                price_df["news_volume"] = 0
                price_df["avg_tone"] = np.nan
                merged = price_df
            else:
                continue
        else:
            # Compute hourly news series
            news_df = pd.DataFrame(articles)
            news_df["date_str"] = news_df["date"].astype(str)
            news_df["hour"] = news_df["date_str"].str[:10]
            news_df["datetime_utc"] = pd.to_datetime(news_df["hour"], format="%Y%m%d%H", errors="coerce")

            hourly = news_df.groupby("datetime_utc").agg(
                news_volume=("gkg_id", "count"),
                avg_tone=("tone", "mean"),
            ).reset_index()

            print(f"    Hourly buckets: {len(hourly)}")

            if price_df is not None:
                # Merge with price - normalize timezones
                price_df["hour"] = price_df["datetime_utc"].dt.tz_localize(None).dt.floor("h")
                hourly["hour"] = hourly["datetime_utc"].dt.tz_localize(None).dt.floor("h")

                merged = pd.merge(price_df, hourly, on="hour", how="outer")
                merged["datetime_utc"] = pd.to_datetime(merged["hour"], utc=True)
                merged = merged.drop(columns=["hour"])
                merged = merged.sort_values("datetime_utc")

                # Fill missing
                merged["news_volume"] = merged["news_volume"].fillna(0).astype(int)
                merged["price"] = merged["price"].interpolate(method="linear", limit_direction="both")
                merged["price"] = merged["price"].ffill().bfill()

                merged = merged[["datetime_utc", "price", "news_volume", "avg_tone"]]
            else:
                merged = hourly.rename(columns={"news_volume": "news_volume", "avg_tone": "avg_tone"})
                merged["price"] = np.nan
                merged = merged[["datetime_utc", "price", "news_volume", "avg_tone"]]

        # Save CSV
        csv_path = output_dir / f"event_{event_id}_{safe_name}_merged_series.csv"
        merged.to_csv(csv_path, index=False)
        print(f"    CSV saved: {csv_path} ({len(merged)} rows)")

        # Generate plot
        png_path = output_dir / f"event_{event_id}_{safe_name}_alignment_result.png"
        plot_alignment(merged, event_title, str(png_path))
        print(f"    PNG saved: {png_path}")

        results.append({
            "event_id": event_id,
            "event_title": event_title,
            "csv": str(csv_path),
            "png": str(png_path),
            "rows": len(merged),
            "news_total": len(articles),
            "search_terms": search_terms,
        })

    return results


def plot_alignment(df, title, output_path):
    """Create a dual-axis alignment plot: price (left) + news_volume/avg_tone (right)."""
    fig, ax1 = plt.subplots(figsize=(16, 8))

    df = df.copy()
    df = df.sort_values("datetime_utc")

    # Price line
    if "price" in df.columns and df["price"].notna().any():
        valid = df[df["price"].notna()]
        ax1.plot(valid["datetime_utc"], valid["price"], color="#2c7fb8", linewidth=1.5,
                 alpha=0.9, label="Polymarket Price")
        ax1.fill_between(valid["datetime_utc"], 0, valid["price"],
                          color="#2c7fb8", alpha=0.1)
    ax1.set_ylabel("Price (USDC)", color="#2c7fb8", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax1.set_ylim(0, 1.05)

    # News volume bars
    ax2 = ax1.twinx()
    if "news_volume" in df.columns and df["news_volume"].sum() > 0:
        non_zero = df[df["news_volume"] > 0]
        ax2.bar(non_zero["datetime_utc"], non_zero["news_volume"],
                width=0.03, color="#ff7f0e", alpha=0.4, label="GDELT News Volume")
    ax2.set_ylabel("News Volume (hourly)", color="#ff7f0e", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    # Avg tone line
    ax3 = ax1.twinx()
    if "avg_tone" in df.columns and df["avg_tone"].notna().any():
        valid_tone = df[df["avg_tone"].notna()]
        # Smooth tone with rolling window
        if len(valid_tone) > 12:
            tone_smoothed = valid_tone.set_index("datetime_utc")["avg_tone"].rolling(12, center=True).mean()
            ax3.plot(tone_smoothed.index, tone_smoothed.values, color="#2ca02c",
                     linewidth=1.0, alpha=0.8, label="Avg Tone (12h smoothed)")
        else:
            ax3.plot(valid_tone["datetime_utc"], valid_tone["avg_tone"], color="#2ca02c",
                     linewidth=1.0, alpha=0.8, label="Avg Tone")
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Avg Tone", color="#2ca02c", fontsize=12)
    ax3.tick_params(axis="y", labelcolor="#2ca02c")

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc="upper left", fontsize=10)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Date (UTC)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved to {output_path}")


def generate_summary(results, output_dir):
    """Generate summary CSV and report markdown."""
    summary_csv = output_dir / "qwen_tech_style_plot_summary_2026-04-21.csv"
    report_md = output_dir / "qwen_tech_style_plot_report_2026-04-21.md"

    # CSV
    rows = []
    for i, r in enumerate(results):
        rows.append({
            "event_id": r["event_id"],
            "event_title": r["event_title"],
            "status": "ok",
            "token_id": "",
            "selected_market_question": r["event_title"],
            "png": r["png"],
            "series_csv": r["csv"],
            "news_total": r["news_total"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(summary_csv, index=False)
    print(f"Summary CSV: {summary_csv}")

    # Report markdown
    with open(report_md, "w") as f:
        f.write("# Knowledge Graph GDELT Event Plots\n\n")
        f.write(f"- Events processed: {len(results)}\n")
        f.write(f"- Knowledge graph: {DATA_DIR}/knowledge_graph.json\n")
        f.write(f"- GDELT source: {DATA_DIR}/gdelt_master.duckdb\n\n")
        f.write("| # | event_id | event_title | entities | news_articles |\n")
        f.write("|---|----------|-------------|----------|---------------|\n")
        for i, r in enumerate(results):
            terms = ", ".join(r.get("search_terms", [])[:5])
            f.write(f"| {i+1} | {r['event_id']} | {r['event_title'][:60]} | {terms} | {r['news_total']:,} |\n")

    print(f"Report: {report_md}")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "result/kg_gdelt_alignment"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Knowledge Graph → GDELT News Alignment")
    print("=" * 60)

    # Load knowledge graph
    print("\nLoading knowledge graph...")
    graph = load_graph_from_file()
    print(f"  Events: {graph['stats']['total_events']}")
    print(f"  Nodes: {graph['stats']['total_nodes']}")
    print(f"  Edges: {graph['stats']['total_edges']}")

    # Attach event titles from entities.json
    entities_path = DATA_DIR / "entities.json"
    if entities_path.exists():
        with open(entities_path) as f:
            entities_data = json.load(f)
        graph["_events_raw"] = entities_data

    # Initialize matcher
    print("\nInitializing GDELT matcher (DuckDB)...")
    matcher = RealGDELTMatcher(graph)

    # Generate merged series
    print("\nGenerating merged series and plots...")
    results = generate_merged_series(graph, matcher, output_dir)

    # Summary
    print("\n" + "=" * 60)
    generate_summary(results, output_dir)

    matcher.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
