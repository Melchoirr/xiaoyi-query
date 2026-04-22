import argparse
import re
import subprocess
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


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


def build_hybrid_panel_if_needed(event_pool_csv: str) -> None:
    cmd = [
        "python",
        "analysis/build_top10_panel_dataset.py",
        "--event-pool-csv",
        event_pool_csv,
        "--retrieval-mode",
        "hybrid",
    ]
    subprocess.run(cmd, check=True)


def setup_cjk_font() -> None:
    """Set a usable CJK font for Chinese labels on different systems."""
    preferred = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Arial Unicode MS",
    ]

    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in preferred:
        if name in installed:
            chosen = name
            break

    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [chosen]
    # Ensure minus sign renders correctly with CJK fonts.
    plt.rcParams["axes.unicode_minus"] = False


def add_future_vol_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.sort_values(["event_id", "datetime_utc"]).copy()
    out = []
    for event_id, g in x.groupby("event_id"):
        y = g.copy()
        # Future realized volatility: sum of abs returns in next H hours.
        y[f"future_rv_t{horizon}"] = (
            y["abs_ret_1h"].shift(-1).rolling(window=horizon, min_periods=horizon).sum()
        )
        out.append(y)
    return pd.concat(out, ignore_index=True)


def event_level_metrics(g: pd.DataFrame, horizon: int, spike_z: float, high_vol_q: float) -> dict:
    x = g.dropna(subset=["news_zscore", f"future_rv_t{horizon}"]).copy()
    if len(x) < 80:
        return {
            "n_obs": len(x),
            "n_spike": 0,
            "p_high_given_spike": np.nan,
            "p_high_given_nonspike": np.nan,
            "uplift_pp": np.nan,
            "uplift_ratio": np.nan,
            "mean_future_rv_spike": np.nan,
            "mean_future_rv_nonspike": np.nan,
            "mannwhitney_p": np.nan,
        }

    x["spike"] = (x["news_zscore"] >= spike_z).astype(int)
    rv_col = f"future_rv_t{horizon}"
    vol_thr = x[rv_col].quantile(high_vol_q)
    x["high_vol"] = (x[rv_col] >= vol_thr).astype(int)

    spike = x[x["spike"] == 1]
    nonspike = x[x["spike"] == 0]
    if len(spike) < 10 or len(nonspike) < 20:
        return {
            "n_obs": len(x),
            "n_spike": len(spike),
            "p_high_given_spike": np.nan,
            "p_high_given_nonspike": np.nan,
            "uplift_pp": np.nan,
            "uplift_ratio": np.nan,
            "mean_future_rv_spike": np.nan,
            "mean_future_rv_nonspike": np.nan,
            "mannwhitney_p": np.nan,
        }

    p1 = float(spike["high_vol"].mean())
    p0 = float(nonspike["high_vol"].mean())
    mean1 = float(spike[rv_col].mean())
    mean0 = float(nonspike[rv_col].mean())

    # One-sided test: future volatility after spike > after non-spike.
    _, p_mw = mannwhitneyu(spike[rv_col], nonspike[rv_col], alternative="greater")

    return {
        "n_obs": len(x),
        "n_spike": len(spike),
        "p_high_given_spike": p1,
        "p_high_given_nonspike": p0,
        "uplift_pp": p1 - p0,
        "uplift_ratio": (p1 / p0) if p0 > 0 else np.nan,
        "mean_future_rv_spike": mean1,
        "mean_future_rv_nonspike": mean0,
        "mannwhitney_p": float(p_mw),
    }


def write_report(path: Path, summary: dict, by_event: pd.DataFrame, settings: dict, chart_prob: Path, chart_dist: Path) -> None:
    lines = [
        "# 假设验证报告：新闻激增是否引发后续剧烈波动",
        "",
        "## 验证目标",
        "",
        "- 假设：当新闻量显著激增时，市场在后续窗口内发生剧烈波动的概率更高。",
        "",
        "## 数据与方法",
        "",
        "- 新闻召回：分层召回（关键词粗召回 + 向量精排 hybrid）。",
        f"- 新闻激增定义：news_zscore >= {settings['spike_z']}",
        f"- 后续波动窗口：未来 {settings['horizon']} 小时累计 abs_ret_1h",
        f"- 剧烈波动阈值：各事件内 future_rv 的 {int(settings['high_vol_q']*100)} 分位数",
        "- 显著性检验：Mann-Whitney U（单边，检验激增组未来波动更高）",
        "",
        "## 总体结果",
        "",
        f"- 样本事件数：{summary['n_event_used']}",
        f"- 总样本数：{summary['n_total']}",
        f"- 激增样本占比：{summary['spike_ratio']:.3f}",
        f"- P(高波动|激增)：{summary['p1']:.3f}",
        f"- P(高波动|非激增)：{summary['p0']:.3f}",
        f"- 概率提升（百分点）：{summary['uplift_pp']:.3f}",
        f"- 概率提升倍数：{summary['uplift_ratio']:.3f}",
        f"- 未来波动均值（激增组）：{summary['mean1']:.4f}",
        f"- 未来波动均值（非激增组）：{summary['mean0']:.4f}",
        f"- Mann-Whitney 单边 p 值：{summary['p_mw']:.6f}",
        "",
        "## 分事件结果",
        "",
        by_event.to_markdown(index=False),
        "",
        "## 可视化",
        "",
        f"- 分事件条件概率图：{chart_prob}",
        f"- 总体分布对比图：{chart_dist}",
        "",
        "## 结论口径",
        "",
        "- 若 P(高波动|激增) 明显高于 P(高波动|非激增)，且 p 值较小，可认为假设得到支持。",
        "- 若提升存在但不显著，则说明新闻有方向性信号但稳定性不足，需要更强特征或更长样本期。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate whether news spikes precede large volatility")
    parser.add_argument("--event-pool-csv", default=None, help="Event pool CSV path; default picks latest result/event_pool_top10_*.csv")
    parser.add_argument("--panel-csv", default=None, help="Panel CSV path; default picks latest result/event_panel_top10_*.csv")
    parser.add_argument("--summary-csv", default=None, help="Summary CSV path; default picks latest result/event_panel_summary_top10_*.csv")
    parser.add_argument("--min-observed-ratio", type=float, default=0.60)
    parser.add_argument("--spike-z", type=float, default=2.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--high-vol-q", type=float, default=0.75)
    args = parser.parse_args()

    event_pool_path = resolve_input_path(args.event_pool_csv, "event_pool_top10_*.csv")

    # Force hybrid retrieval panel for this hypothesis validation.
    build_hybrid_panel_if_needed(str(event_pool_path))

    panel_path = resolve_input_path(args.panel_csv, "event_panel_top10_*.csv")
    summary_path = resolve_input_path(args.summary_csv, "event_panel_summary_top10_*.csv")
    date_tag = infer_date_tag(panel_path)

    setup_cjk_font()

    panel = pd.read_csv(panel_path)
    summary = pd.read_csv(summary_path)
    panel["datetime_utc"] = pd.to_datetime(panel["datetime_utc"], utc=True)
    panel["event_id"] = panel["event_id"].astype(str)
    summary["event_id"] = summary["event_id"].astype(str)

    keep_ids = summary[
        (summary["observed_ratio"] >= args.min_observed_ratio)
        & (summary["low_match_flag"] == 0)
    ]["event_id"]
    work = panel[panel["event_id"].isin(set(keep_ids))].copy()
    work = add_future_vol_targets(work, horizon=args.horizon)

    rows = []
    pooled_parts = []
    rv_col = f"future_rv_t{args.horizon}"
    for event_id, g in work.groupby("event_id"):
        event_title = g["event_title"].iloc[0]
        m = event_level_metrics(g, args.horizon, args.spike_z, args.high_vol_q)
        rows.append({"event_id": event_id, "event_title": event_title, **m})

        x = g.dropna(subset=["news_zscore", rv_col]).copy()
        x["spike"] = (x["news_zscore"] >= args.spike_z).astype(int)
        x["high_vol"] = (x[rv_col] >= x[rv_col].quantile(args.high_vol_q)).astype(int)
        pooled_parts.append(x[["spike", "high_vol", rv_col]])

    by_event = pd.DataFrame(rows).sort_values("uplift_pp", ascending=False)
    pooled = pd.concat(pooled_parts, ignore_index=True)
    spike = pooled[pooled["spike"] == 1]
    non = pooled[pooled["spike"] == 0]

    p1 = float(spike["high_vol"].mean()) if len(spike) else np.nan
    p0 = float(non["high_vol"].mean()) if len(non) else np.nan
    mean1 = float(spike[rv_col].mean()) if len(spike) else np.nan
    mean0 = float(non[rv_col].mean()) if len(non) else np.nan
    _, p_mw = mannwhitneyu(spike[rv_col], non[rv_col], alternative="greater")

    out_dir = Path("result") / "by_date" / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_event_csv = out_dir / f"news_spike_hypothesis_by_event_{date_tag}.csv"
    out_report_md = out_dir / f"news_spike_hypothesis_report_{date_tag}.md"
    out_prob_png = out_dir / f"news_spike_hypothesis_prob_{date_tag}.png"
    out_dist_png = out_dir / f"news_spike_hypothesis_dist_{date_tag}.png"

    by_event.to_csv(out_event_csv, index=False)

    # Plot 1: conditional probability uplift by event.
    plot_df = by_event.dropna(subset=["p_high_given_spike", "p_high_given_nonspike"]).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["event_id"].astype(str), plot_df["p_high_given_nonspike"], alpha=0.6, label="P(高波动|非激增)")
    ax.bar(plot_df["event_id"].astype(str), plot_df["p_high_given_spike"], alpha=0.6, label="P(高波动|激增)")
    ax.set_ylabel("条件概率")
    ax.set_xlabel("事件ID")
    ax.set_title("分事件：新闻激增对后续高波动概率的影响")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_prob_png, dpi=220)
    plt.close(fig)

    # Plot 2: pooled distribution comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([non[rv_col].dropna(), spike[rv_col].dropna()], tick_labels=["非激增", "激增"], showfliers=False)
    ax.set_title("总体：未来波动分布对比")
    ax.set_ylabel(f"future_rv_t{args.horizon}")
    plt.tight_layout()
    plt.savefig(out_dist_png, dpi=220)
    plt.close(fig)

    summary_obj = {
        "n_event_used": int(work["event_id"].nunique()),
        "n_total": int(len(pooled)),
        "spike_ratio": float(pooled["spike"].mean()),
        "p1": p1,
        "p0": p0,
        "uplift_pp": float(p1 - p0),
        "uplift_ratio": float(p1 / p0) if p0 > 0 else np.nan,
        "mean1": mean1,
        "mean0": mean0,
        "p_mw": float(p_mw),
    }

    write_report(
        out_report_md,
        summary_obj,
        by_event,
        {
            "spike_z": args.spike_z,
            "horizon": args.horizon,
            "high_vol_q": args.high_vol_q,
        },
        out_prob_png,
        out_dist_png,
    )

    print(f"Input event pool: {event_pool_path}")
    print(f"Input panel: {panel_path}")
    print(f"Input summary: {summary_path}")
    print(f"Saved event metrics: {out_event_csv}")
    print(f"Saved report: {out_report_md}")
    print(f"Saved probability chart: {out_prob_png}")
    print(f"Saved distribution chart: {out_dist_png}")
    print(summary_obj)


if __name__ == "__main__":
    main()