import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_threshold_data(study_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_fp = study_dir / "threshold_study_summary_2026-04-21.csv"
    if not summary_fp.exists():
        # fallback: pick any threshold_study_summary_*.csv
        files = sorted(study_dir.glob("threshold_study_summary_*.csv"))
        if not files:
            raise FileNotFoundError(f"No threshold summary found under {study_dir}")
        summary_fp = files[-1]

    summary = pd.read_csv(summary_fp).sort_values("threshold")

    rows = []
    for th_dir in sorted(p for p in study_dir.glob("threshold_*") if p.is_dir()):
        tag = th_dir.name.replace("threshold_", "")
        th_value = float(tag.replace("p", "."))

        ranking_files = sorted(th_dir.glob("kept_ranking_by_event_*_th_*.csv"))
        if not ranking_files:
            continue

        rk = pd.read_csv(ranking_files[-1])
        if rk.empty:
            continue

        for _, r in rk.iterrows():
            rows.append(
                {
                    "threshold": th_value,
                    "event_id": str(r["event_id"]),
                    "event_title": str(r["event_title"]),
                    "kept_news_count": int(r["kept_news_count"]),
                }
            )

    event_df = pd.DataFrame(rows)
    return summary, event_df


def plot_overview(summary: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    sns.set_style("whitegrid")

    p1 = out_dir / "qwen_threshold_total_kept_news.png"
    p2 = out_dir / "qwen_threshold_events_with_news.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["threshold"], summary["total_kept_news"], marker="o", linewidth=2)
    ax.set_title("Qwen Threshold vs Total Kept News")
    ax.set_xlabel("semantic_min_similarity")
    ax.set_ylabel("total_kept_news")
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(p1, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["threshold"], summary["events_with_kept_news"], marker="o", linewidth=2, color="tab:orange")
    ax.set_title("Qwen Threshold vs Events With News")
    ax.set_xlabel("semantic_min_similarity")
    ax.set_ylabel("events_with_kept_news")
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(p2, dpi=220)
    plt.close(fig)

    return p1, p2


def plot_event_heatmap(event_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if event_df.empty:
        return None

    pivot = event_df.pivot_table(index="event_title", columns="threshold", values="kept_news_count", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

    p = out_dir / "qwen_threshold_event_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, max(4, 0.8 * len(pivot.index))))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, ax=ax)
    ax.set_title("Qwen Kept News Count Heatmap by Event and Threshold")
    ax.set_xlabel("threshold")
    ax.set_ylabel("event_title")
    plt.tight_layout()
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def write_report(summary: pd.DataFrame, event_df: pd.DataFrame, out_dir: Path, figs: list[Path]) -> Path:
    out_md = out_dir / "qwen_visualization_report_2026-04-21.md"

    best_cov = summary.loc[summary["events_with_kept_news"].idxmax()]
    best_volume = summary.loc[summary["total_kept_news"].idxmax()]

    lines = [
        "# Qwen 阈值可视化报告",
        "",
        "## 核心结论",
        "",
        f"- 覆盖事件数最高阈值: {best_cov['threshold']} (events_with_kept_news={int(best_cov['events_with_kept_news'])})",
        f"- 召回新闻量最高阈值: {best_volume['threshold']} (total_kept_news={int(best_volume['total_kept_news'])})",
        "- 从本次曲线看，阈值降低会显著提高召回量，但低阈值噪声风险同步增加。",
        "",
        "## 图表文件",
        "",
    ]

    for fp in figs:
        if fp is not None:
            lines.append(f"- {fp}")

    lines += [
        "",
        "## 阈值摘要表",
        "",
        summary.sort_values("threshold", ascending=False).to_markdown(index=False),
        "",
    ]

    if not event_df.empty:
        top = (
            event_df.groupby("event_title", as_index=False)["kept_news_count"]
            .sum()
            .sort_values("kept_news_count", ascending=False)
            .head(10)
        )
        lines += [
            "## 事件累计召回 Top10",
            "",
            top.to_markdown(index=False),
            "",
        ]

    lines += [
        "## 相关性计算说明",
        "",
        "- 当前 Qwen panel 中部分事件在分析窗口内 price 变化或 news_zscore 方差为 0，Pearson 相关性会退化为 NaN。",
        "- 因此本次先交付稳定可复现的阈值-召回可视化结果；后续如需相关性图，可先筛选有有效方差的事件再单独跑 lead-lag 分析。",
    ]

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization package for Qwen threshold study")
    parser.add_argument(
        "--study-dir",
        default="result/by_date/2026-04-21/threshold_study_qwen3_2026-04-21",
        help="Path to qwen threshold study directory",
    )
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    if not study_dir.exists():
        raise FileNotFoundError(f"Study dir not found: {study_dir}")

    out_dir = study_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, event_df = read_threshold_data(study_dir)
    p1, p2 = plot_overview(summary, out_dir)
    p3 = plot_event_heatmap(event_df, out_dir)
    md = write_report(summary, event_df, out_dir, [p1, p2, p3])

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    if p3:
        print(f"Saved: {p3}")
    print(f"Saved: {md}")


if __name__ == "__main__":
    main()
