"""Plot full ETT dataset curves with train/val/test spans marked."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
SPAN_STYLE = {
    "train": ("#dbeafe", 0.28),
    "val": ("#fef3c7", 0.34),
    "test": ("#dcfce7", 0.32),
    "not used": ("#e5e7eb", 0.28),
}


def _split_borders(data_path, n_rows):
    name = os.path.splitext(os.path.basename(data_path))[0].lower()
    unit = 4 if name.startswith("ettm") else 1
    train = 12 * 30 * 24 * unit
    val = 4 * 30 * 24 * unit
    test = 4 * 30 * 24 * unit
    test_end = min(train + val + test, n_rows)
    borders = {
        "train": (0, train),
        "val": (train, train + val),
        "test": (train + val, test_end),
    }
    if test_end < n_rows:
        borders["not used"] = (test_end, n_rows)
    return borders


def _span_end_date(dates, end):
    if end < len(dates):
        return dates.iloc[end]
    return dates.iloc[-1]


def _add_split_spans(ax, dates, borders):
    n = len(dates)
    for name, (start, end) in borders.items():
        if start >= n:
            continue
        color, alpha = SPAN_STYLE[name]
        ax.axvspan(dates.iloc[start], _span_end_date(dates, min(end, n - 1)), color=color, alpha=alpha)
        if end < n:
            ax.axvline(dates.iloc[end], color="#374151", linestyle="--", linewidth=0.8, alpha=0.7)


def _normalize_by_train(df, borders):
    out = df.copy()
    train_start, train_end = borders["train"]
    mean = out[CHANNELS].iloc[train_start:train_end].mean()
    std = out[CHANNELS].iloc[train_start:train_end].std(ddof=0).replace(0.0, 1.0)
    out[CHANNELS] = (out[CHANNELS] - mean) / std
    return out


def _plot_dataset(csv_path, output_dir, normalize):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    missing = [col for col in CHANNELS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    borders = _split_borders(csv_path, len(df))
    plot_df = _normalize_by_train(df, borders) if normalize else df
    data_name = os.path.splitext(os.path.basename(csv_path))[0]
    suffix = "train_normalized" if normalize else "raw"

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(17, 13.5), sharex=True)
    dates = plot_df["date"]
    for ax, col in zip(axes, CHANNELS):
        _add_split_spans(ax, dates, borders)
        ax.plot(dates, plot_df[col], color="#111827", linewidth=0.35)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Date")

    legend_handles = [
        Patch(facecolor=SPAN_STYLE["train"][0], alpha=SPAN_STYLE["train"][1], label="train"),
        Patch(facecolor=SPAN_STYLE["val"][0], alpha=SPAN_STYLE["val"][1], label="val"),
        Patch(facecolor=SPAN_STYLE["test"][0], alpha=SPAN_STYLE["test"][1], label="test"),
    ]
    if "not used" in borders:
        legend_handles.append(
            Patch(
                facecolor=SPAN_STYLE["not used"][0],
                alpha=SPAN_STYLE["not used"][1],
                label="not used",
            )
        )
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=False)
    title_mode = "train-normalized" if normalize else "raw"
    split_text = (
        f"train [0,{borders['train'][1]}), val [{borders['val'][0]},{borders['val'][1]}), "
        f"test [{borders['test'][0]},{borders['test'][1]})"
    )
    if "not used" in borders:
        split_text += f", not used [{borders['not used'][0]},{borders['not used'][1]})"
    fig.suptitle(f"{data_name} {title_mode} full curves | {split_text}", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.965])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{data_name.lower()}_{suffix}_split_curves.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="dataset")
    parser.add_argument("--data_paths", default="ETTh2.csv,ETTm2.csv")
    parser.add_argument("--output_dir", default="outputs/figures/data_analysis")
    parser.add_argument("--normalized_only", action="store_true")
    parser.add_argument("--raw_only", action="store_true")
    args = parser.parse_args()

    data_paths = [item.strip() for item in args.data_paths.split(",") if item.strip()]
    if args.normalized_only and args.raw_only:
        raise ValueError("Choose at most one of --normalized_only and --raw_only")

    modes = []
    if not args.normalized_only:
        modes.append(False)
    if not args.raw_only:
        modes.append(True)

    for data_path in data_paths:
        csv_path = os.path.join(args.root_path, data_path)
        for normalize in modes:
            out_path = _plot_dataset(csv_path, args.output_dir, normalize=normalize)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
