#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/zhaodawei/miniconda3/envs/stocks_env/bin/python}"
EXP="experiments/future_factor_retrieval_etth1/decomposed_trend_season_retrieval.py"
OUT_ROOT="outputs/trend_season_decomp_etth1"
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

# decomp_kernel, component_ratio_min, component_ratio_max, tag
CONFIGS=(
  "13 0.5 2.0 k13_r05_20"
  "25 0.5 2.0 k25_r05_20"
  "49 0.5 2.0 k49_r05_20"
  "25 0.25 4.0 k25_r025_40"
  "25 0.75 1.5 k25_r075_15"
)

for pred_len in "${PRED_LENS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    read -r kernel ratio_min ratio_max tag <<< "${config}"
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
      --decomp_kernel "${kernel}" \
      --candidate_ks 400,200,100,200 \
      --shape_weight 1.0 \
      --raw_weight 0.25 \
      --time_weight 0.05 \
      --future_weight 0.03 \
      --std_weight 0.25 \
      --trend_top_k 50 \
      --season_top_k 50 \
      --trend_align clipped_std \
      --season_align clipped_std \
      --component_ratio_min "${ratio_min}" \
      --component_ratio_max "${ratio_max}" \
      --std_eps 1e-6 \
      --weight_mode inverse \
      --output_dir "${output_dir}" \
      --append_summary "${SUMMARY}"
  done
done

"${PYTHON_BIN}" scripts/trend_season_decomp_etth1/summarize.py "${SUMMARY}"
