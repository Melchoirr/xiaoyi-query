import argparse
import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
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


def setup_cjk_font() -> None:
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
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def add_future_rv(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = []
    for eid, g in df.groupby("event_id"):
        x = g.sort_values("datetime_utc").copy()
        x[f"future_rv_t{horizon}"] = x["abs_ret_1h"].shift(-1).rolling(window=horizon, min_periods=horizon).sum()
        out.append(x)
    return pd.concat(out, ignore_index=True)


def make_event_timeline_plot(g: pd.DataFrame, event_id: str, event_title: str, horizon: int, spike_z: float, out_path: Path) -> None:
    rv_col = f"future_rv_t{horizon}"
    x = g.sort_values("datetime_utc").copy()
    x = x.dropna(subset=[rv_col])

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    # 1) price
    axes[0].plot(x["datetime_utc"], x["price"], color="tab:blue", lw=1.6)
    axes[0].set_ylabel("价格")
    axes[0].set_title(f"事件 {event_id}: {event_title}")

    # 2) news volume vs immediate volatility (abs_ret_1h)
    ax2 = axes[1]
    ax2.bar(x["datetime_utc"], x["news_volume"], color="tab:gray", alpha=0.35, width=0.03, label="新闻量")
    ax2.set_ylabel("新闻量")
    ax2b = ax2.twinx()
    ax2b.plot(x["datetime_utc"], x["abs_ret_1h"], color="tab:green", lw=1.2, label="当期波动 abs_ret_1h")
    ax2b.set_ylabel("当期波动")
    ax2b.grid(False)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")

    # 3) news_zscore with spike markers
    axes[2].plot(x["datetime_utc"], x["news_zscore"], color="tab:orange", lw=1.2)
    axes[2].axhline(spike_z, color="red", linestyle="--", alpha=0.8, label=f"激增阈值 z={spike_z}")
    spike_df = x[x["news_zscore"] >= spike_z]
    if not spike_df.empty:
        axes[2].scatter(spike_df["datetime_utc"], spike_df["news_zscore"], s=18, color="red", alpha=0.8, label="激增点")
    axes[2].set_ylabel("新闻z分数")
    axes[2].legend(loc="upper right")

    # 4) future realized volatility
    axes[3].plot(x["datetime_utc"], x[rv_col], color="tab:green", lw=1.4)
    axes[3].set_ylabel(f"未来{horizon}h波动")
    axes[3].set_xlabel("时间(UTC)")

    # Overlay spike vertical lines for visual lead-lag checking.
    for t in spike_df["datetime_utc"].head(120):
        axes[3].axvline(t, color="red", alpha=0.08, lw=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def make_event_study_curve(df: pd.DataFrame, spike_z: float, horizon_before: int, horizon_after: int, out_path: Path) -> None:
    """Aggregate average abs_ret around spike times for eyeballing dynamic response."""
    records = []
    for eid, g in df.groupby("event_id"):
        x = g.sort_values("datetime_utc").reset_index(drop=True)
        spike_idx = x.index[x["news_zscore"] >= spike_z].tolist()
        for idx in spike_idx:
            for k in range(-horizon_before, horizon_after + 1):
                j = idx + k
                if 0 <= j < len(x):
                    records.append({"k": k, "abs_ret_1h": x.at[j, "abs_ret_1h"]})

    if not records:
        return

    ev = pd.DataFrame(records)
    agg = ev.groupby("k", as_index=False).agg(mean_abs_ret=("abs_ret_1h", "mean"))

    plt.figure(figsize=(10, 4.8))
    plt.plot(agg["k"], agg["mean_abs_ret"], marker="o", color="tab:purple")
    plt.axvline(0, color="black", linestyle="--", alpha=0.8)
    plt.title("新闻激增事件窗：平均波动响应")
    plt.xlabel("相对新闻激增时点的小时偏移")
    plt.ylabel("平均abs_ret_1h")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate time-wise charts for news spikes and volatility")
    parser.add_argument("--panel-csv", default=None, help="Panel CSV path; default picks latest result/event_panel_top10_*.csv")
    parser.add_argument("--summary-csv", default=None, help="Summary CSV path; default picks latest result/event_panel_summary_top10_*.csv")
    parser.add_argument("--spike-z", type=float, default=2.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--min-observed-ratio", type=float, default=0.60)
    args = parser.parse_args()

    panel_path = resolve_input_path(args.panel_csv, "event_panel_top10_*.csv")
    summary_path = resolve_input_path(args.summary_csv, "event_panel_summary_top10_*.csv")
    date_tag = infer_date_tag(panel_path)

    setup_cjk_font()

    panel = pd.read_csv(panel_path)
    summary = pd.read_csv(summary_path)
    panel["datetime_utc"] = pd.to_datetime(panel["datetime_utc"], utc=True)
    panel["event_id"] = panel["event_id"].astype(str)
    summary["event_id"] = summary["event_id"].astype(str)

    keep = summary[
        (summary["observed_ratio"] >= args.min_observed_ratio)
        & (summary["low_match_flag"] == 0)
    ]["event_id"]
    work = panel[panel["event_id"].isin(set(keep))].copy()
    work = add_future_rv(work, horizon=args.horizon)

    out_dir = Path("result") / "by_date" / date_tag / f"timewise_news_volatility_charts_{date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for eid, g in work.groupby("event_id"):
        title = g["event_title"].iloc[0]
        out_path = out_dir / f"event_{eid}_timewise.png"
        make_event_timeline_plot(
            g,
            event_id=eid,
            event_title=title,
            horizon=args.horizon,
            spike_z=args.spike_z,
            out_path=out_path,
        )

    event_study_path = Path("result") / "by_date" / date_tag / f"timewise_event_study_curve_{date_tag}.png"
    make_event_study_curve(
        work,
        spike_z=args.spike_z,
        horizon_before=12,
        horizon_after=24,
        out_path=event_study_path,
    )

    print(f"Input panel: {panel_path}")
    print(f"Input summary: {summary_path}")
    print(f"Saved per-event charts dir: {out_dir}")
    print(f"Saved event-study curve: {event_study_path}")
    print(f"Events plotted: {work['event_id'].nunique()}")


if __name__ == "__main__":
    main()