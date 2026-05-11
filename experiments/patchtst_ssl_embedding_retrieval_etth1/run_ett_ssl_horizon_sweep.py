"""
Run PatchTST SSL embedding retrieval on ETT hourly datasets and horizons.

Defaults:
  datasets: ETTh1,ETTh2
  pred_lens: 96,192,336,720
  seq_len: 192
  embedding_weight: 0.3
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "experiments" / "patchtst_ssl_embedding_retrieval_etth1" / "compare_patchtst_ssl_embedding_retrieval.py"


def _parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _run(cmd):
    print("\n" + "=" * 80)
    print(" ".join(str(part) for part in cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--datasets", type=str, default="ETTh1,ETTh2")
    parser.add_argument("--pred_lens", type=str, default="96,192,336,720")
    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--flags", type=str, default="val,test")
    parser.add_argument("--embedding_weight", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--retrieval_batch_size", type=int, default=256)
    parser.add_argument("--align_mode", type=str, default="std", choices=["std", "mean"])
    parser.add_argument("--ssl_epochs", type=int, default=10)
    parser.add_argument("--ssl_batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output_root", type=str, default="outputs/patchtst_ssl_embedding_retrieval_ett_sweep")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=None,
        help="optional root containing prior per-run checkpoints to reuse",
    )
    args = parser.parse_args()

    datasets = _parse_csv(args.datasets)
    pred_lens = _parse_csv(args.pred_lens, int)
    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = REPO_ROOT / args.checkpoint_root if args.checkpoint_root else None

    summary_paths = []
    for data in datasets:
        data_path = f"{data}.csv"
        for pred_len in pred_lens:
            run_dir = output_root / f"PatchTSTSSLFactorRetrieval_{data}_M_sl{args.seq_len}_pl{pred_len}_ew{args.embedding_weight:g}"
            checkpoint_path = None
            if checkpoint_root is not None:
                checkpoint_path = (
                    checkpoint_root
                    / f"PatchTSTSSLFactorRetrieval_{data}_M_sl{args.seq_len}_pl{pred_len}_ew{args.embedding_weight:g}"
                    / "checkpoints"
                    / "patchtst_ssl.pth"
                )
            cmd = [
                args.python_bin,
                "-u",
                str(SCRIPT),
                "--root_path",
                "dataset",
                "--data_path",
                data_path,
                "--seq_len",
                str(args.seq_len),
                "--pred_len",
                str(pred_len),
                "--flags",
                args.flags,
                "--methods",
                "shape,shape_ssl_embedding",
                "--embedding_weights",
                str(args.embedding_weight),
                "--top_k",
                str(args.top_k),
                "--retrieval_batch_size",
                str(args.retrieval_batch_size),
                "--align_mode",
                args.align_mode,
                "--ssl_epochs",
                str(args.ssl_epochs),
                "--ssl_batch_size",
                str(args.ssl_batch_size),
                "--device",
                args.device,
                "--output_dir",
                str(run_dir),
            ]
            if checkpoint_path is not None:
                cmd.extend(["--checkpoint_path", str(checkpoint_path)])
            _run(cmd)
            summary_paths.append(run_dir / "summary.csv")

    rows = []
    for summary_path in summary_paths:
        df = pd.read_csv(summary_path)
        for row in df.to_dict("records"):
            rows.append(
                {
                    "data": row["setting"].split("_")[1],
                    "seq_len": args.seq_len,
                    "pred_len": int(row["setting"].split("_sl")[1].split("_pl")[1]),
                    **row,
                }
            )

    combined_path = output_root / "summary_all.csv"
    if rows:
        with open(combined_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\ncombined summary saved to: {combined_path}")


if __name__ == "__main__":
    main()
