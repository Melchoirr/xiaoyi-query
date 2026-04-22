import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run single-event Qwen correlation plots for all tech events")
    parser.add_argument(
        "--event-pool-csv",
        default="result/by_date/2026-04-21/event_pool_top10_2026-04-21_tech.csv",
    )
    parser.add_argument(
        "--retrieval-dir",
        default="result/by_date/2026-04-21/news_retrieval_by_event_2026-04-21",
    )
    parser.add_argument(
        "--runner-script",
        default="analysis/plot_single_event_qwen_correlation.py",
    )
    args = parser.parse_args()

    pool_fp = Path(args.event_pool_csv)
    retrieval_dir = Path(args.retrieval_dir)
    runner = Path(args.runner_script)

    if not pool_fp.exists():
        raise FileNotFoundError(f"event pool not found: {pool_fp}")
    if not retrieval_dir.exists():
        raise FileNotFoundError(f"retrieval dir not found: {retrieval_dir}")
    if not runner.exists():
        raise FileNotFoundError(f"runner script not found: {runner}")

    pool = pd.read_csv(pool_fp)
    event_ids = [str(x) for x in pool["event_id"].tolist()]

    ok_rows = []
    fail_rows = []

    for eid in event_ids:
        retrieval_fp = retrieval_dir / f"event_{eid}_retrieval_ranked_2026-04-21.csv"
        if not retrieval_fp.exists():
            fail_rows.append({"event_id": eid, "status": "missing_retrieval_csv", "detail": str(retrieval_fp)})
            continue

        cmd = [
            "python",
            str(runner),
            "--retrieval-csv",
            str(retrieval_fp),
            "--max-lag",
            "24",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            fail_rows.append(
                {
                    "event_id": eid,
                    "status": "failed",
                    "detail": (p.stderr or p.stdout or "")[-2000:],
                }
            )
            continue

        out_dir = Path("result") / "by_date" / "2026-04-21" / f"single_event_qwen_{eid}"
        report_fp = out_dir / f"single_event_report_{eid}.md"

        corr_keep = None
        corr_all = None
        best_lag = None
        best_lag_corr = None
        if report_fp.exists():
            txt = report_fp.read_text(encoding="utf-8")
            m1 = re.search(r"corr\(abs_ret_1h, kept_news_volume\):\s*([^\n]+)", txt)
            m2 = re.search(r"corr\(abs_ret_1h, all_news_volume\):\s*([^\n]+)", txt)
            m3 = re.search(r"best_lag_by_abs_corr:\s*([^\n]+)", txt)
            m4 = re.search(r"best_lag_corr:\s*([^\n]+)", txt)
            corr_keep = m1.group(1).strip() if m1 else ""
            corr_all = m2.group(1).strip() if m2 else ""
            best_lag = m3.group(1).strip() if m3 else ""
            best_lag_corr = m4.group(1).strip() if m4 else ""

        ok_rows.append(
            {
                "event_id": eid,
                "event_title": str(pool.loc[pool["event_id"].astype(str) == eid, "event_title"].iloc[0]),
                "status": "ok",
                "output_dir": str(out_dir),
                "corr_keep": corr_keep,
                "corr_all": corr_all,
                "best_lag": best_lag,
                "best_lag_corr": best_lag_corr,
            }
        )

    out_root = Path("result") / "by_date" / "2026-04-21" / "qwen_all_tech_event_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    ok_df = pd.DataFrame(ok_rows)
    fail_df = pd.DataFrame(fail_rows)
    ok_csv = out_root / "qwen_all_tech_event_summary_2026-04-21.csv"
    fail_csv = out_root / "qwen_all_tech_event_failures_2026-04-21.csv"
    ok_df.to_csv(ok_csv, index=False)
    fail_df.to_csv(fail_csv, index=False)

    lines = [
        "# Qwen All Tech Events Correlation Batch",
        "",
        f"- Event pool: {pool_fp}",
        f"- Retrieval dir: {retrieval_dir}",
        f"- Success events: {len(ok_df)}",
        f"- Failed events: {len(fail_df)}",
        "",
        "## Success Summary",
        "",
    ]

    if not ok_df.empty:
        lines.append(ok_df.to_markdown(index=False))
    else:
        lines.append("No successful event outputs.")

    lines += ["", "## Failure Summary", ""]
    if not fail_df.empty:
        lines.append(fail_df.to_markdown(index=False))
    else:
        lines.append("No failures.")

    lines += ["", "## Per-event folders", ""]
    for _, r in ok_df.iterrows():
        lines.append(f"- {r['output_dir']}")

    md = out_root / "qwen_all_tech_event_report_2026-04-21.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {ok_csv}")
    print(f"Saved: {fail_csv}")
    print(f"Saved: {md}")
    print(f"success={len(ok_df)} fail={len(fail_df)}")


if __name__ == "__main__":
    main()
