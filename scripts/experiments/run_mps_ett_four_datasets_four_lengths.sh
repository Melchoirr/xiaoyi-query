#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/zhaodawei/miniconda3/envs/agent/bin/python}"
EXP="experiments/future_factor_retrieval_etth1/channel_independent_direction_retrieval_standalone.py"
ROOT_DIR="outputs/mps_ett_four_datasets_four_lengths"
CHANNEL_BACKEND="${CHANNEL_BACKEND:-batched}"
SELECTED_BATCH_SIZE="${SELECTED_BATCH_SIZE:-32}"

mkdir -p "${ROOT_DIR}"

for data_path in ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv; do
  data_name="${data_path%.csv}"
  for pred_len in 96 192 336 720; do
    out_dir="${ROOT_DIR}/${data_name}_sl96_pl${pred_len}"
    "${PYTHON_BIN}" "${EXP}" \
      --root_path dataset \
      --data_path "${data_path}" \
      --features M \
      --target OT \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "${pred_len}" \
      --flag test \
      --weights 1.5 \
      --top_k 50 \
      --backend mps \
      --channel_backend "${CHANNEL_BACKEND}" \
      --selected_batch_size "${SELECTED_BATCH_SIZE}" \
      --output_dir "${out_dir}"
  done
done
