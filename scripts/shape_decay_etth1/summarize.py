import csv
import sys
from collections import defaultdict


def fmt(value):
    return f"{float(value):.6f}"


def config_id(row):
    return f"d{float(row['shape_decay']):g}"


def main(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    test_rerank = [
        row for row in rows
        if row["flag"] == "test" and row["method"] == "rerank"
    ]
    if not test_rerank:
        raise SystemExit(f"no test rerank rows found in {path}")

    by_pred = defaultdict(list)
    for row in test_rerank:
        by_pred[int(row["pred_len"])].append(row)

    print("\nBest ETTh1 shape-decay test rerank by pred_len")
    print("| pred_len | shape_decay | MSE | MAE | baseline MSE | delta MSE vs shape |")
    print("|---:|---:|---:|---:|---:|---:|")
    for pred_len in sorted(by_pred):
        best = min(by_pred[pred_len], key=lambda row: float(row["mse"]))
        print(
            f"| {pred_len} | {config_id(best)} | {fmt(best['mse'])} | "
            f"{fmt(best['mae'])} | {fmt(best['baseline_mse'])} | "
            f"{fmt(best['delta_mse_vs_shape'])} |"
        )

    print("\nAll ETTh1 shape-decay test rerank rows")
    print("| pred_len | shape_decay | MSE | MAE | baseline MSE | delta MSE vs shape |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in sorted(test_rerank, key=lambda item: (int(item["pred_len"]), float(item["shape_decay"]))):
        print(
            f"| {row['pred_len']} | {config_id(row)} | {fmt(row['mse'])} | "
            f"{fmt(row['mae'])} | {fmt(row['baseline_mse'])} | "
            f"{fmt(row['delta_mse_vs_shape'])} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python summarize.py outputs/shape_decay_etth1/summary_all.csv")
    main(sys.argv[1])
