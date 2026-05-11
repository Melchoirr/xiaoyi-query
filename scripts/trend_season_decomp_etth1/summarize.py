import csv
import sys
from collections import defaultdict


def fmt(value):
    return f"{float(value):.6f}"


def config_id(row):
    ratio_min = row.get("component_ratio_min") or row.get("season_ratio_min")
    ratio_max = row.get("component_ratio_max") or row.get("season_ratio_max")
    return (
        f"k{int(row['decomp_kernel'])}_"
        f"r{float(ratio_min):g}_{float(ratio_max):g}"
    )


def main(path):
    rows = [
        row for row in csv.DictReader(open(path, newline="", encoding="utf-8"))
        if row["flag"] == "test"
    ]
    if not rows:
        raise SystemExit(f"no test rows found in {path}")

    by_pred = defaultdict(list)
    for row in rows:
        by_pred[int(row["pred_len"])].append(row)

    print("\nBest ETTh1 trend/season retrieval by pred_len")
    print("| pred_len | config | MSE | MAE |")
    print("|---:|---|---:|---:|")
    for pred_len in sorted(by_pred):
        best = min(by_pred[pred_len], key=lambda row: float(row["mse"]))
        print(f"| {pred_len} | {config_id(best)} | {fmt(best['mse'])} | {fmt(best['mae'])} |")

    print("\nAll ETTh1 trend/season retrieval rows")
    print("| pred_len | config | MSE | MAE | trend_dist_mean | season_dist_mean |")
    print("|---:|---|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: (int(item["pred_len"]), config_id(item))):
        print(
            f"| {row['pred_len']} | {config_id(row)} | {fmt(row['mse'])} | "
            f"{fmt(row['mae'])} | {fmt(row['trend_dist_mean'])} | "
            f"{fmt(row['season_dist_mean'])} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python summarize.py outputs/trend_season_decomp_etth1/summary_all.csv")
    main(sys.argv[1])
