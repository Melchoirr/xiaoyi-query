import csv
import sys


def fmt(value):
    return f"{float(value):.6f}"


def main(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    test = [row for row in rows if row["flag"] == "test" and row["method"] == "rerank"]
    if not test:
        raise SystemExit(f"no test rerank rows found in {path}")

    print("\nETTh1 full-rerank test results")
    print("| pred_len | MSE | MAE | shape baseline MSE | delta MSE vs shape | candidate_mean |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in sorted(test, key=lambda item: int(item["pred_len"])):
        print(
            f"| {row['pred_len']} | {fmt(row['mse'])} | {fmt(row['mae'])} | "
            f"{fmt(row['baseline_mse'])} | {fmt(row['delta_mse_vs_shape'])} | "
            f"{float(row['candidate_mean']):.1f} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python summarize.py outputs/full_rerank_etth1/summary_all.csv")
    main(sys.argv[1])
