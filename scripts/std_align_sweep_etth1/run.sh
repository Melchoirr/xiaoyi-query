#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/zhaodawei/miniconda3/envs/stocks_env/bin/python}"
EXP="experiments/future_factor_retrieval_etth1/simple_align_retrieval.py"
OUT_ROOT="outputs/std_align_sweep_etth1"
SUMMARY="${OUT_ROOT}/summary_all.csv"

mkdir -p "${OUT_ROOT}"
rm -f "${SUMMARY}"

PRED_LENS=(96 192 336 720)

# std_weight, std_ratio_min, std_ratio_max, tag
CONFIGS=(
  "0.00 0.5 2.0 no_std_penalty_r05_20"
  "0.10 0.5 2.0 sw010_r05_20"
  "0.25 0.5 2.0 sw025_r05_20"
  "0.50 0.5 2.0 sw050_r05_20"
  "0.25 0.25 4.0 sw025_r025_40"
  "0.25 0.75 1.5 sw025_r075_15"
  "0.50 0.25 4.0 sw050_r025_40"
  "0.50 0.75 1.5 sw050_r075_15"
)

for pred_len in "${PRED_LENS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    read -r std_weight ratio_min ratio_max tag <<< "${config}"
    output_dir="${OUT_ROOT}/ETTh1_sl192_pl${pred_len}_${tag}"
    done_file="${output_dir}/summary.csv"

    if [[ -s "${done_file}" ]]; then
      echo "skip existing ${output_dir}"
      continue
    fi

    "${PYTHON_BIN}" "${EXP}" \
      --root_path dataset \
      --data_path ETTh1.csv \
      --freq h \
      --seq_len 192 \
      --pred_len "${pred_len}" \
      --flags val,test \
      --candidate_ks 400,200,100,200 \
      --top_k 50 \
      --shape_weight 1.0 \
      --raw_weight 0.25 \
      --time_weight 0.05 \
      --future_weight 0.03 \
      --std_weight "${std_weight}" \
      --align_mode clipped_std \
      --std_eps 1e-6 \
      --std_ratio_min "${ratio_min}" \
      --std_ratio_max "${ratio_max}" \
      --output_dir "${output_dir}" \
      --append_summary "${SUMMARY}"
  done
done

"${PYTHON_BIN}" scripts/std_align_sweep_etth1/summarize.py "${SUMMARY}"
