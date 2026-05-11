#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/zhaodawei/miniconda3/envs/stocks_env/bin/python}"
EXP="experiments/future_factor_retrieval_etth1/simple_align_retrieval_shape_decay.py"
OUT_ROOT="outputs/shape_decay_etth1"
SUMMARY="${OUT_ROOT}/summary_all.csv"

mkdir -p "${OUT_ROOT}"
rm -f "${SUMMARY}"

append_csv() {
  local src="$1"
  if [[ ! -s "${src}" ]]; then
    return
  fi
  if [[ ! -s "${SUMMARY}" ]]; then
    cat "${src}" >> "${SUMMARY}"
  else
    tail -n +2 "${src}" >> "${SUMMARY}"
  fi
}

PRED_LENS=(96 192 336 720)
DECAYS=(1.0 0.995 0.99 0.985 0.98 0.97)

for pred_len in "${PRED_LENS[@]}"; do
  for decay in "${DECAYS[@]}"; do
    tag="d${decay//./p}"
    output_dir="${OUT_ROOT}/ETTh1_sl192_pl${pred_len}_${tag}"
    done_file="${output_dir}/summary.csv"

    if [[ -s "${done_file}" ]]; then
      echo "skip existing ${output_dir}"
      append_csv "${done_file}"
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
      --shape_decay "${decay}" \
      --shape_weight 1.0 \
      --raw_weight 0.25 \
      --time_weight 0.05 \
      --future_weight 0.03 \
      --std_weight 0.25 \
      --align_mode clipped_std \
      --std_eps 1e-6 \
      --std_ratio_min 0.25 \
      --std_ratio_max 4.0 \
      --output_dir "${output_dir}" \
      --append_summary "${SUMMARY}"
  done
done

"${PYTHON_BIN}" scripts/shape_decay_etth1/summarize.py "${SUMMARY}"
