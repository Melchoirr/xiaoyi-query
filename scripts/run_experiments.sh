#!/usr/bin/env bash
# =====================================================================
# run_experiments.sh - Time-Series Forecasting Baseline Parameter Sweep
# =====================================================================
# v3.4: --use_gpu flag, V100 32G optimized parameters
# =====================================================================
set -euo pipefail

# ── Project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

# ── Default parameters (V100 32G optimized) ───────────────────────────
MODELS="${MODELS:-PatternSearch,LSHSearch,SAXSearch,DTWSearch,MatrixProfileSearch,TS2VecSearch,RAGSearch}"
SEQ_LENS="${SEQ_LENS:-96}"
PRED_LENS="${PRED_LENS:-96}"
REVIN_TYPES="${REVIN_TYPES:-dual}"
TOP_K_VALUES="${TOP_K_VALUES:-5}"
GPU_FLAG=""
PARALLEL_FLAG=""
DRY_RUN=false

# ── Argument parsing ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --models)       MODELS="$2"; shift 2 ;;
        --revin-types)  REVIN_TYPES="$2"; shift 2 ;;
        --seq-lens)     SEQ_LENS="$2"; shift 2 ;;
        --pred-lens)    PRED_LENS="$2"; shift 2 ;;
        --top-k)        TOP_K_VALUES="$2"; shift 2 ;;
        --parallel)     PARALLEL_FLAG="--parallel"; shift ;;
        --use-gpu)      GPU_FLAG="--use_gpu"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Split comma-separated strings into arrays ─────────────────────────
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra SEQ_ARR <<< "$SEQ_LENS"
IFS=',' read -ra PRED_ARR <<< "$PRED_LENS"
IFS=',' read -ra REVIN_ARR <<< "$REVIN_TYPES"
IFS=',' read -ra TOPK_ARR <<< "$TOP_K_VALUES"

# ── Compute total experiment count ─────────────────────────────────────
TOTAL=$((${#MODEL_ARR[@]} * ${#SEQ_ARR[@]} * ${#PRED_ARR[@]} * ${#REVIN_ARR[@]} * ${#TOPK_ARR[@]}))

echo "========================================================"
echo "Time-Series Forecasting Baseline - Parameter Sweep  [v3.4]"
echo "========================================================"
echo "Models:      ${MODEL_ARR[*]}"
echo "seq_len:     ${SEQ_ARR[*]}"
echo "pred_len:    ${PRED_ARR[*]}"
echo "revin_type:  ${REVIN_ARR[*]}"
echo "top_k:       ${TOPK_ARR[*]}"
echo "GPU:         ${GPU_FLAG:-disabled}"
echo "Parallel:    ${PARALLEL_FLAG:-disabled}"
echo "Total:       $TOTAL experiments"
echo "========================================================"

# ── Main loop ─────────────────────────────────────────────────────────
RUN_PY="$PROJECT_DIR/run.py"

COUNT=0
PASS=0
FAIL=0

for model in "${MODEL_ARR[@]}"; do
    for seq in "${SEQ_ARR[@]}"; do
        for pred in "${PRED_ARR[@]}"; do
            for revin in "${REVIN_ARR[@]}"; do
                for k in "${TOPK_ARR[@]}"; do
                    ((++COUNT))

                    revin_suffix=""
                    if [[ "$revin" != "none" ]]; then
                        char=$(echo "${revin:0:1}" | tr '[:upper:]' '[:lower:]')
                        revin_suffix="_R${char}"
                    fi

                    exp_id="${model}_seq${seq}_pred${pred}_k${k}${revin_suffix}"

                    echo ""
                    echo "[$COUNT/$TOTAL] Launching: $exp_id"
                    echo "--------------------------------------------------------------------------"

                    # Build command
                    cmd=(
                        "$PYTHON" "$RUN_PY"
                        --model "$model"
                        --seq_len "$seq"
                        --pred_len "$pred"
                        --revin_type "$revin"
                        --top_k "$k"
                    )

                    # V100 32G optimized defaults are baked into run.py defaults
                    # Pass GPU and parallel flags if set
                    [[ -n "$GPU_FLAG" ]]       && cmd+=("$GPU_FLAG")
                    [[ -n "$PARALLEL_FLAG" ]] && cmd+=("$PARALLEL_FLAG")

                    if [[ "$DRY_RUN" == true ]]; then
                        echo "DRY RUN: ${cmd[*]}"
                        continue
                    fi

                    # Execute and tee log (timestamped by run.py)
                    set +e
                    "${cmd[@]}" 2>&1 | tee "$exp_id.tmp.log"
                    EXIT_CODE=${PIPESTATUS[0]}
                    set -e

                    # run.py creates run_{timestamp}/ under results/
                    # Move the per-experiment tee log to the latest run_ dir if exists
                    latest_run=$(find "$PROJECT_DIR/results" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort -r | head -1)
                    if [[ -n "$latest_run" && -f "$exp_id.tmp.log" ]]; then
                        mv "$exp_id.tmp.log" "$latest_run/logs/${exp_id}.log"
                    fi

                    if [[ "$EXIT_CODE" -eq 0 ]]; then
                        ((++PASS))
                        echo "  [PASS]"
                    else
                        ((++FAIL))
                        echo "  [FAIL] Exit code: $EXIT_CODE"
                    fi

                    echo "Progress: $COUNT/$TOTAL | PASS: $PASS | FAIL: $FAIL"
                    sleep 0.2
                done
            done
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Sweep complete!"
echo "PASS: $PASS / $TOTAL"
echo "FAIL: $FAIL / $TOTAL"
echo "========================================================"

# ── Auto-generate summary from the latest run_ dir ────────────────────
latest_run=$(find "$PROJECT_DIR/results" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort -r | head -1)
if [[ -n "$latest_run" && -f "$latest_run/summary_metrics.csv" ]]; then
    SUMMARY_CSV="$latest_run/summary_metrics.csv"
    echo ""
    echo "Generating summary analysis from: $SUMMARY_CSV"
    "$PYTHON" - "$SUMMARY_CSV" "$latest_run" <<'PYEOF'
import sys, os
import pandas as pd

csv_path = sys.argv[1]
run_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path)

try:
    df = pd.read_csv(csv_path)
    if 'MAE' in df.columns:
        df_ok = df[df['status'] == 'success'].copy()
        if not df_ok.empty:
            df_ok = df_ok.sort_values('MAE')
            print("\n=== MAE Ranking (Top 10) ===")
            cols = ['model', 'revin_type', 'seq_len', 'pred_len', 'MAE']
            present_cols = [c for c in cols if c in df_ok.columns]
            print(df_ok[present_cols].head(10).to_string(index=False))

            # Generate bar chart
            try:
                from plotting import plot_summary_bar
                bar_path = os.path.join(run_dir, 'summary_MAE_bar.png')
                plot_summary_bar(csv_path, metric='MAE', save_path=bar_path)
                print(f"\nBar chart saved: {bar_path}")
            except ImportError:
                print("\nplotting.py not found, skipping bar chart.")
except Exception as e:
    print(f"Error during analysis: {e}")

print(f"\nResults in: {run_dir}/")
PYEOF
else
    echo "No summary_metrics.csv found in $latest_run, skipping analysis."
fi

echo "All experiment logs in: $latest_run/logs/"
