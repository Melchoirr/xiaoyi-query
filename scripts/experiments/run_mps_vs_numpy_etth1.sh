#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/zhaodawei/miniconda3/envs/agent/bin/python"
EXP="experiments/future_factor_retrieval_etth1/channel_independent_direction_retrieval_standalone.py"
CHANNEL_BACKEND="${CHANNEL_BACKEND:-batched}"
SELECTED_BATCH_SIZE="${SELECTED_BATCH_SIZE:-32}"
COMMON_ARGS=(
  --root_path dataset
  --data_path ETTh1.csv
  --features M
  --target OT
  --seq_len 96
  --label_len 48
  --pred_len 96
  --flag test
  --weights 1.5
  --top_k 50
)

"${PYTHON_BIN}" "${EXP}" "${COMMON_ARGS[@]}" --backend mps --channel_backend "${CHANNEL_BACKEND}" --selected_batch_size "${SELECTED_BATCH_SIZE}" --output_dir outputs/mps_check_etth1
"${PYTHON_BIN}" "${EXP}" "${COMMON_ARGS[@]}" --backend numpy --output_dir outputs/numpy_check_etth1

"${PYTHON_BIN}" - <<'PY'
import csv
from pathlib import Path

def read_row(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]

mps = read_row(Path("outputs/mps_check_etth1/summary.csv"))
cpu = read_row(Path("outputs/numpy_check_etth1/summary.csv"))
print("mps_mse=", mps["mse"], "numpy_mse=", cpu["mse"], "delta=", float(mps["mse"]) - float(cpu["mse"]))
print("mps_mae=", mps["mae"], "numpy_mae=", cpu["mae"], "delta=", float(mps["mae"]) - float(cpu["mae"]))
PY
