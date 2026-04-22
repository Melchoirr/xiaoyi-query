import argparse
import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


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


def compute_lag_corr(vol: pd.Series, news: pd.Series, max_lag: int = 24) -> pd.DataFrame:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        corr = vol.corr(news.shift(lag))
        rows.append({"lag": lag, "corr": float(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def run_event_regression(df_event: pd.DataFrame, best_lag: int) -> dict:
    x = df_event.copy()
    x["ar1"] = x["abs_ret_1h"].shift(1)
    x["news_lagged"] = x["news_zscore"].shift(best_lag)

    reg = x[["abs_ret_1h", "ar1", "news_lagged"]].dropna()
    if len(reg) < 48:
        return {
            "reg_nobs": len(reg),
            "beta_news": np.nan,
            "pvalue_news": np.nan,
            "r2": np.nan,
        }

    y = reg["abs_ret_1h"]
    X = sm.add_constant(reg[["ar1", "news_lagged"]], has_constant="add")
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    return {
        "reg_nobs": int(model.nobs),
        "beta_news": float(model.params.get("news_lagged", np.nan)),
        "pvalue_news": float(model.pvalues.get("news_lagged", np.nan)),
        "r2": float(model.rsquared),
    }


def plot_event_figure(df_event: pd.DataFrame, lag_corr: pd.DataFrame, event_id: str, event_title: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

    ax1.plot(df_event["datetime_utc"], df_event["price"], color="tab:blue", lw=1.8, label="price")
    ax1.set_ylabel("Price")
    ax1.set_title(f"Event {event_id}: {event_title}")

    ax1b = ax1.twinx()
    ax1b.bar(df_event["datetime_utc"], df_event["news_volume"], color="tab:gray", alpha=0.25, width=0.03, label="news_volume")
    ax1b.set_ylabel("News Volume")
    ax1b.grid(False)

    ax2.bar(lag_corr["lag"], lag_corr["corr"], color="tab:orange", alpha=0.8)
    ax2.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Lag (hour)")
    ax2.set_ylabel("Corr(abs_ret_1h, news_zscore.shift(lag))")
    ax2.set_title("Lead-Lag Correlation")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_report(
    report_path: Path,
    panel_path: Path,
    summary_path: Path,
    metrics_df: pd.DataFrame,
    pooled: dict,
    quality_filtered_count: int,
    total_events: int,
    chart_dir: Path,
    heatmap_path: Path,
) -> None:
    lines = []
    lines.append("# Top10 Panel Alignment Study Report")
    lines.append("")
    lines.append(f"- Input panel: {panel_path}")
    lines.append(f"- Input summary: {summary_path}")
    lines.append(f"- Events analyzed: {quality_filtered_count}/{total_events}")
    lines.append("- Method: hourly lead-lag correlation with max lag 24 and HAC-robust AR(1)+news regression")
    lines.append("")
    lines.append("## Pooled Findings")
    lines.append("")
    lines.append(f"- Mean best-lag absolute correlation: {pooled['mean_abs_corr']:.4f}")
    lines.append(f"- Events with news-leading pattern (best_lag > 0): {pooled['news_lead_count']}")
    lines.append(f"- Events with market-leading pattern (best_lag < 0): {pooled['market_lead_count']}")
    lines.append(f"- Events with significant news beta (p < 0.10): {pooled['sig_beta_count']}")
    lines.append("")
    lines.append("## Event Metrics")
    lines.append("")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Event charts directory: {chart_dir}")
    lines.append(f"- Lag-correlation heatmap: {heatmap_path}")
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- Positive best_lag means news tends to lead volatility by that many hours.")
    lines.append("- Negative best_lag means market volatility tends to lead observed news intensity.")
    lines.append("- Because panel uses forward-filled prices between trades, low observed_ratio events should be down-weighted in final model selection.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alignment study from unified top10 panel csv")
    parser.add_argument("--panel-csv", default=None, help="Panel CSV path; default picks latest result/event_panel_top10_*.csv")
    parser.add_argument("--summary-csv", default=None, help="Summary CSV path; default picks latest result/event_panel_summary_top10_*.csv")
    parser.add_argument("--min-observed-ratio", type=float, default=0.60)
    parser.add_argument("--max-lag", type=int, default=24)
    args = parser.parse_args()

    panel_path = resolve_input_path(args.panel_csv, "event_panel_top10_*.csv")
    summary_path = resolve_input_path(args.summary_csv, "event_panel_summary_top10_*.csv")
    date_tag = infer_date_tag(panel_path)
    out_dir = Path("result") / "by_date" / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_path = out_dir / f"alignment_metrics_top10_{date_tag}.csv"
    out_report_path = out_dir / f"alignment_report_top10_{date_tag}.md"
    out_heatmap_path = out_dir / f"alignment_lag_heatmap_top10_{date_tag}.png"
    chart_dir = out_dir / f"alignment_event_charts_{date_tag}"
    chart_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    summary = pd.read_csv(summary_path)
    panel["datetime_utc"] = pd.to_datetime(panel["datetime_utc"], utc=True)

    keep_events = summary[
        (summary["observed_ratio"] >= args.min_observed_ratio)
        & (summary["low_match_flag"] == 0)
    ]["event_id"].astype(str)

    panel["event_id"] = panel["event_id"].astype(str)
    work = panel[panel["event_id"].isin(set(keep_events))].copy()

    metrics_rows = []
    lag_rows = []

    for event_id, g in work.groupby("event_id"):
        g = g.sort_values("datetime_utc").copy()
        event_title = g["event_title"].iloc[0]

        g = g.dropna(subset=["abs_ret_1h", "news_zscore"])
        if len(g) < 48:
            continue

        lag_corr = compute_lag_corr(g["abs_ret_1h"], g["news_zscore"], max_lag=args.max_lag)
        lag_corr["event_id"] = event_id
        lag_rows.append(lag_corr)

        usable = lag_corr.dropna(subset=["corr"])
        if usable.empty:
            continue
        best_idx = usable["corr"].abs().idxmax()
        best_lag = int(usable.loc[best_idx, "lag"])
        best_corr = float(usable.loc[best_idx, "corr"])

        reg_result = run_event_regression(g, best_lag=best_lag)

        metrics_rows.append(
            {
                "event_id": event_id,
                "event_title": event_title,
                "n_hours": int(len(g)),
                "best_lag": best_lag,
                "best_corr": best_corr,
                "abs_best_corr": abs(best_corr),
                **reg_result,
            }
        )

        fig_path = chart_dir / f"event_{event_id}.png"
        plot_event_figure(g, lag_corr, event_id, event_title, fig_path)

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise RuntimeError("No event passed minimum data requirements after filtering.")

    metrics_df = metrics_df.sort_values("abs_best_corr", ascending=False)
    metrics_df.to_csv(out_metrics_path, index=False)

    lag_df = pd.concat(lag_rows, ignore_index=True)
    heat = lag_df.pivot_table(index="event_id", columns="lag", values="corr", aggfunc="mean")
    plt.figure(figsize=(14, 6))
    sns.heatmap(heat, cmap="coolwarm", center=0, vmin=-0.4, vmax=0.4)
    plt.title("Lag Correlation Heatmap by Event")
    plt.xlabel("Lag (hour)")
    plt.ylabel("Event ID")
    plt.tight_layout()
    plt.savefig(out_heatmap_path, dpi=220)
    plt.close()

    pooled = {
        "mean_abs_corr": float(metrics_df["abs_best_corr"].mean()),
        "news_lead_count": int((metrics_df["best_lag"] > 0).sum()),
        "market_lead_count": int((metrics_df["best_lag"] < 0).sum()),
        "sig_beta_count": int((metrics_df["pvalue_news"] < 0.10).sum()),
    }

    write_report(
        report_path=out_report_path,
        panel_path=panel_path,
        summary_path=summary_path,
        metrics_df=metrics_df,
        pooled=pooled,
        quality_filtered_count=int(work["event_id"].nunique()),
        total_events=int(summary["event_id"].nunique()),
        chart_dir=chart_dir,
        heatmap_path=out_heatmap_path,
    )

    print(f"Input panel: {panel_path}")
    print(f"Input summary: {summary_path}")
    print(f"Saved metrics: {out_metrics_path}")
    print(f"Saved report: {out_report_path}")
    print(f"Saved heatmap: {out_heatmap_path}")
    print(f"Saved event charts in: {chart_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()