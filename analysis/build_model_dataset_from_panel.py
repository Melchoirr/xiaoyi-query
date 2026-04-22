import argparse
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def pick_latest_result_file(pattern: str) -> Path:
    files = list(Path("result").glob(f"**/{pattern}"))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: result/{pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_input_path(cli_value: str | None, pattern: str) -> Path:
    if cli_value:
        p = Path(cli_value)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p
    return pick_latest_result_file(pattern)


def infer_date_tag(path: Path) -> str:
    m = DATE_RE.search(path.name)
    return m.group(1) if m else date.today().isoformat()


def add_features_and_targets(g: pd.DataFrame, max_lag: int, horizons: list[int]) -> pd.DataFrame:
    x = g.sort_values("datetime_utc").copy()

    # Basic calendar features.
    x["hour"] = x["datetime_utc"].dt.hour
    x["dow"] = x["datetime_utc"].dt.dayofweek
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)
    x["dow_sin"] = np.sin(2 * np.pi * x["dow"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["dow"] / 7)

    # Lag features.
    base_cols = ["ret_1h", "abs_ret_1h", "news_zscore", "news_volume", "avg_tone", "price"]
    for col in base_cols:
        for lag in range(0, max_lag + 1):
            name = f"{col}_lag{lag}"
            x[name] = x[col].shift(lag)

    # Future targets for forecasting.
    # Direction next 1h: classification target in {-1, 0, 1}.
    x["target_dir_t1"] = np.sign(x["ret_1h"].shift(-1)).astype("float")

    for h in horizons:
        # Future realized volatility (sum of absolute returns in next h hours).
        x[f"target_rv_t{h}"] = (
            x["abs_ret_1h"].shift(-1).rolling(window=h, min_periods=h).sum()
        )
        # Future cumulative return in next h hours.
        x[f"target_cumret_t{h}"] = (
            x["ret_1h"].shift(-1).rolling(window=h, min_periods=h).sum()
        )

    return x


def assign_splits(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("datetime_utc")

    t0 = out["datetime_utc"].min()
    t1 = out["datetime_utc"].max()
    span = (t1 - t0).total_seconds()

    train_cut = t0 + pd.to_timedelta(span * train_ratio, unit="s")
    val_cut = t0 + pd.to_timedelta(span * (train_ratio + val_ratio), unit="s")

    out["split"] = np.where(
        out["datetime_utc"] <= train_cut,
        "train",
        np.where(out["datetime_utc"] <= val_cut, "val", "test"),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready supervised dataset from alignment panel")
    parser.add_argument("--panel-csv", default=None, help="Panel CSV path; default picks latest result/event_panel_top10_*.csv")
    parser.add_argument("--summary-csv", default=None, help="Summary CSV path; default picks latest result/event_panel_summary_top10_*.csv")
    parser.add_argument("--min-observed-ratio", type=float, default=0.60)
    parser.add_argument("--max-lag", type=int, default=24)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    panel_path = resolve_input_path(args.panel_csv, "event_panel_top10_*.csv")
    summary_path = resolve_input_path(args.summary_csv, "event_panel_summary_top10_*.csv")
    date_tag = infer_date_tag(panel_path)

    panel = pd.read_csv(panel_path)
    summary = pd.read_csv(summary_path)
    panel["datetime_utc"] = pd.to_datetime(panel["datetime_utc"], utc=True)
    panel["event_id"] = panel["event_id"].astype(str)
    summary["event_id"] = summary["event_id"].astype(str)

    horizons = [int(v) for v in args.horizons.split(",") if v.strip()]

    # Quality filter to avoid noisy pseudo-series.
    keep_ids = summary[
        (summary["observed_ratio"] >= args.min_observed_ratio)
        & (summary["low_match_flag"] == 0)
    ]["event_id"]

    work = panel[panel["event_id"].isin(set(keep_ids))].copy()

    parts = []
    for event_id, g in work.groupby("event_id"):
        feat = add_features_and_targets(g, max_lag=args.max_lag, horizons=horizons)
        feat["event_id"] = event_id
        parts.append(feat)

    ds = pd.concat(parts, ignore_index=True)
    ds = assign_splits(ds, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # Keep only rows with complete feature/target set.
    lag_cols = [c for c in ds.columns if "_lag" in c]
    target_cols = [c for c in ds.columns if c.startswith("target_")]
    required = lag_cols + target_cols + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    ds = ds.dropna(subset=required)

    # Sort for deterministic training pipelines.
    ds = ds.sort_values(["event_id", "datetime_utc"]).reset_index(drop=True)

    out_dir = Path("result") / "by_date" / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"model_dataset_top10_{date_tag}.csv"
    out_stats = out_dir / f"model_dataset_stats_top10_{date_tag}.csv"
    out_doc = out_dir / f"model_dataset_notes_top10_{date_tag}.md"

    ds.to_csv(out_csv, index=False)

    stats = (
        ds.groupby(["split", "event_id"], as_index=False)
        .agg(
            n_rows=("event_id", "size"),
            mean_abs_ret=("abs_ret_1h", "mean"),
            mean_news_z=("news_zscore", "mean"),
        )
        .sort_values(["split", "event_id"])
    )
    stats.to_csv(out_stats, index=False)

    lines = [
        "# Model Dataset Notes",
        "",
        f"- Source panel: {panel_path}",
        f"- Source summary: {summary_path}",
        f"- Event filter: observed_ratio >= {args.min_observed_ratio} and low_match_flag == 0",
        f"- Max lag: {args.max_lag}",
        f"- Horizons: {horizons}",
        f"- Split ratio train/val/test: {args.train_ratio}/{args.val_ratio}/{1 - args.train_ratio - args.val_ratio}",
        f"- Dataset file: {out_csv}",
        f"- Stats file: {out_stats}",
        "",
        "## Main Feature Families",
        "",
        "- Price/return lags: ret_1h_lag*, abs_ret_1h_lag*, price_lag*",
        "- News lags: news_zscore_lag*, news_volume_lag*, avg_tone_lag*",
        "- Calendar cyclic features: hour_sin, hour_cos, dow_sin, dow_cos",
        "",
        "## Targets",
        "",
        "- target_dir_t1: next-1h direction sign",
        "- target_rv_tH: next-Hh realized volatility (sum abs ret)",
        "- target_cumret_tH: next-Hh cumulative return (sum ret)",
        "",
    ]
    out_doc.write_text("\n".join(lines), encoding="utf-8")

    print(f"Input panel: {panel_path}")
    print(f"Input summary: {summary_path}")
    print(f"Saved dataset: {out_csv}")
    print(f"Saved stats: {out_stats}")
    print(f"Saved notes: {out_doc}")
    print(f"Rows: {len(ds)}, Events: {ds['event_id'].nunique()}")
    print(stats.groupby('split', as_index=False)['n_rows'].sum().to_string(index=False))


if __name__ == "__main__":
    main()