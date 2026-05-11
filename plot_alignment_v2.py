"""Generate clean dual-axis alignment plots from V2 enhanced alignment CSVs.
Price (line) + News Volume (bars). No tone/sentiment.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Event ID → readable short label for plot title
EVENT_LABELS = {
    "118172": "Will Trump acquire Greenland before 2027?",
    "236884": "Iran x Israel/US conflict ends by...?",
    "237306": "Will Iran strike gulf oil facilities by March 31?",
    "257313": "US-Iran nuclear deal by April 30?",
    "34044": "Will China invade Taiwan by end of 2026?",
    "34050": "Russia x Ukraine ceasefire by end of 2026?",
    "35908": "Who will Trump nominate as Fed Chair?",
    "67284": "Fed decision in March?",
    "73130": "Will the U.S. invade Iran before 2027?",
    "75478": "Fed decision in April?",
}


def load_v2_csv(csv_path):
    """Load a V2 merged series CSV, handling duplicate datetime columns."""
    df = pd.read_csv(csv_path)

    # V2 CSVs may have datetime_utc_x, datetime_utc_y, datetime_utc from merge
    # Prefer the clean 'datetime_utc' column if it exists and is complete
    dt_col = None
    for col in ["datetime_utc", "datetime_utc_x"]:
        if col in df.columns:
            dt_col = col
            break

    if dt_col is None:
        raise ValueError(f"No datetime column found in {csv_path}")

    df["dt"] = pd.to_datetime(df[dt_col], utc=True)
    df = df.sort_values("dt")

    # Price: interpolate gaps
    price = df["price"].interpolate().ffill().bfill() if "price" in df.columns else pd.Series([np.nan]*len(df))

    # News volume
    vol = df["news_volume"].fillna(0) if "news_volume" in df.columns else pd.Series([0]*len(df))

    return pd.DataFrame({"datetime_utc": df["dt"], "price": price, "news_volume": vol})


def plot_event(df, title, output_path):
    """Dual-axis plot: price line (left) + news volume bars (right). No tone."""
    fig, ax1 = plt.subplots(figsize=(16, 7))

    # Price line — auto-scale y-axis to show variation
    valid_price = df[df["price"].notna()]
    if len(valid_price) > 0:
        ax1.plot(valid_price["datetime_utc"], valid_price["price"],
                 color="#2c7fb8", linewidth=1.5, alpha=0.9, label="Polymarket Price")
        p_min = valid_price["price"].min()
        p_max = valid_price["price"].max()
        pad = max((p_max - p_min) * 0.15, 0.005)
        y_bottom = max(0, p_min - pad)
        y_top = p_max + pad
        ax1.fill_between(valid_price["datetime_utc"], y_bottom, valid_price["price"],
                          color="#2c7fb8", alpha=0.08)
        ax1.set_ylim(y_bottom, y_top)
    ax1.set_ylabel("Price (USDC)", color="#2c7fb8", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")

    # News volume bars
    ax2 = ax1.twinx()
    non_zero = df[df["news_volume"] > 0]
    if len(non_zero) > 0:
        ax2.bar(non_zero["datetime_utc"], non_zero["news_volume"],
                width=0.03, color="#ff7f0e", alpha=0.35, label="GDELT News Volume (V2 enhanced)")
    ax2.set_ylabel("News Volume (hourly)", color="#ff7f0e", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Date (UTC)", fontsize=12)
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "result/by_date/2026-05-11/kg_gdelt_alignment_v2")
    output_dir = input_dir  # Save plots alongside CSVs

    csv_files = sorted(input_dir.glob("event_*_merged_series.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Generating V2 alignment plots ({len(csv_files)} events)")
    print("=" * 60)

    for csv_path in csv_files:
        event_id = csv_path.stem.split("_")[1]
        label = EVENT_LABELS.get(event_id, csv_path.stem)
        print(f"\n  {event_id}: {label[:60]}")

        df = load_v2_csv(csv_path)
        print(f"    {len(df)} rows, price range [{df['price'].min():.3f}–{df['price'].max():.3f}], "
              f"news max={df['news_volume'].max():.0f}")

        png_name = csv_path.stem.replace("_merged_series", "_alignment_v2.png")
        plot_event(df, label, output_dir / png_name)

    print(f"\nDone. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
