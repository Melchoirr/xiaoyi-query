#!/usr/bin/env bash
# =====================================================================
# run_experiments.sh - Time-Series Forecasting Baseline  [v3.5]
# =====================================================================
# Smart routing: each model type gets its own parameter grid.
# Single GLOBAL_RUN_DIR shared across all run.py calls (no folder flood).
# =====================================================================
set -euo pipefail

# ── Project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

# ── One-time timestamp directory (shared by all experiments in this run)
GLOBAL_RUN_DIR="${PROJECT_DIR}/results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$GLOBAL_RUN_DIR/logs"

# ── Default sweep parameters ──────────────────────────────────────────
MODELS="${MODELS:-PatternSearch,LSHSearch,SAXSearch,DTWSearch,MatrixProfileSearch,TS2VecSearch,RAGSearch}"
SEQ_LENS="${SEQ_LENS:-96}"
PRED_LENS="${PRED_LENS:-96}"
REVIN_TYPES="${REVIN_TYPES:-dual}"
TOP_K_VALUES="${TOP_K_VALUES:-5}"
PARALLEL_FLAG=""
GPU_FLAG=""
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────
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

IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra SEQ_ARR <<< "$SEQ_LENS"
IFS=',' read -ra PRED_ARR <<< "$PRED_LENS"
IFS=',' read -ra REVIN_ARR <<< "$REVIN_TYPES"
IFS=',' read -ra TOPK_ARR <<< "$TOP_K_VALUES"

# ── Header ─────────────────────────────────────────────────────────────
echo "========================================================"
echo "Time-Series Forecasting Baseline - Smart Sweep  [v3.5]"
echo "========================================================"
echo "GLOBAL_RUN_DIR: $GLOBAL_RUN_DIR"
echo "Models:      ${MODEL_ARR[*]}"
echo "seq_len:     ${SEQ_ARR[*]}"
echo "pred_len:    ${PRED_ARR[*]}"
echo "revin_type:  ${REVIN_ARR[*]}"
echo "top_k vals:  ${TOPK_ARR[*]}"
echo "GPU:         ${GPU_FLAG:-disabled}"
echo "Parallel:    ${PARALLEL_FLAG:-disabled}"
echo "========================================================"

# ── Helper: build the base command array (shared by all models) ───────
build_base_cmd() {
    local seq="$1"
    local pred="$2"
    local revin="$3"
    local run_dir="$4"

    local cmd=(
        "$PYTHON" "$PROJECT_DIR/run.py"
        --seq_len "$seq"
        --pred_len "$pred"
        --revin_type "$revin"
        --run_dir "$run_dir"
    )
    [[ -n "$GPU_FLAG" ]]       && cmd+=("$GPU_FLAG")
    [[ -n "$PARALLEL_FLAG" ]] && cmd+=("$PARALLEL_FLAG")
    echo "${cmd[@]}"
}

# ── Main sweep loop ────────────────────────────────────────────────────
RUN_PY="$PROJECT_DIR/run.py"
COUNT=0; PASS=0; FAIL=0

for model in "${MODEL_ARR[@]}"; do

    # ── Smart routing: resolve this model's parameter arrays ──────────
    # Persistent subshell so array assignments don't leak between models
    source /dev/stdin <<'STAB' 2>/dev/null || true
    case "$model" in

        PatternSearch|LSHSearch|SAXSearch|DTWSearch|MatrixProfileSearch|TS2VecSearch)
            _SEQ_ARR=("${SEQ_ARR[@]}")
            _PRED_ARR=("${PRED_ARR[@]}")
            _REV_ARR=("${REV_ARR[@]}")
            _K_ARR=("${TOPK_ARR[@]}")
            ;;

        RAGSearch)
            # RAGSearch does not use top_k — fix to a single placeholder value
            _SEQ_ARR=("${SEQ_ARR[@]}")
            _PRED_ARR=("${PRED_ARR[@]}")
            _REV_ARR=("${REVIN_ARR[@]}")
            _K_ARR=("__rag__")
            ;;

        *)
            _SEQ_ARR=("${SEQ_ARR[@]}")
            _PRED_ARR=("${PRED_ARR[@]}")
            _REV_ARR=("${REVIN_ARR[@]}")
            _K_ARR=("${TOPK_ARR[@]}")
            ;;
    esac
STAB

    # Fallback if sourcing failed (POSIX compat): manually assign arrays
    if [[ -z "${_SEQ_ARR[*]:-}" ]]; then
        case "$model" in
            RAGSearch)
                _SEQ_ARR=("${SEQ_ARR[@]}")
                _PRED_ARR=("${PRED_ARR[@]}")
                _REV_ARR=("${REVIN_ARR[@]}")
                _K_ARR=("__rag__")
                ;;
            *)
                _SEQ_ARR=("${SEQ_ARR[@]}")
                _PRED_ARR=("${PRED_ARR[@]}")
                _REV_ARR=("${REVIN_ARR[@]}")
                _K_ARR=("${TOPK_ARR[@]}")
                ;;
        esac
    fi

    for seq in "${_SEQ_ARR[@]}"; do
        for pred in "${_PRED_ARR[@]}"; do
            for revin in "${_REV_ARR[@]}"; do
                for k in "${_K_ARR[@]}"; do
                    ((++COUNT))

                    # ── Build revin suffix ─────────────────────────────
                    revin_suffix=""
                    if [[ "$revin" != "none" ]]; then
                        revin_suffix="_R${revin:0:1}"
                        revin_suffix="${revin_suffix,,}"
                    fi

                    # ── Build model params for this run ────────────────
                    # RAGSearch has no top_k; TS2Vec / KNN models have top_k
                    model_params=(
                        --model "$model"
                    )
                    # Initialize k_disp to avoid unbound variable error
                    k_disp=""
                    case "$model" in
                        RAGSearch)
                            # RAGSearch: no --top_k at all
                            k_disp="5"
                            ;;
                        TS2VecSearch|MatrixProfileSearch|DTWSearch|PatternSearch|LSHSearch|SAXSearch)
                            # Strip leading zeros from $k for placeholder "5" etc.
                            [[ "$k" == "__rag__" ]] && k_disp="5" || k_disp="$k"
                            model_params+=(--top_k "$k_disp")
                            ;;
                    esac

                    exp_id="${model}_seq${seq}_pred${pred}_k${k_disp}${revin_suffix}"
                    exp_id="${exp_id//__rag__/5}"   # normalise placeholder back to "5" in name

                    echo ""
                    echo "[$COUNT] Model=$model seq=$seq pred=$pred revin=$revin k=$k"
                    echo "--------------------------------------------------------------------------"

                    # ── Build full command ─────────────────────────────
                    base_cmd=(
                        "$PYTHON" "$PROJECT_DIR/run.py"
                        --model "$model"
                        --seq_len "$seq"
                        --pred_len "$pred"
                        --revin_type "$revin"
                        --run_dir "$GLOBAL_RUN_DIR"
                    )
                    [[ -n "$GPU_FLAG" ]]       && base_cmd+=("$GPU_FLAG")
                    [[ -n "$PARALLEL_FLAG" ]] && base_cmd+=("$PARALLEL_FLAG")

                    # Append top_k only if the model needs it
                    if [[ "$model" != "RAGSearch" ]]; then
                        [[ "$k" != "__rag__" ]] && base_cmd+=(--top_k "$k") \
                                               || base_cmd+=(--top_k 5)
                    fi

                    if [[ "$DRY_RUN" == true ]]; then
                        echo "DRY RUN: ${base_cmd[*]}"
                        continue
                    fi

                    # ── Execute ───────────────────────────────────────
                    log_file="$GLOBAL_RUN_DIR/logs/${exp_id}.log"
                    set +e
                    "${base_cmd[@]}" > "$log_file" 2>&1
                    EXIT_CODE=$?
                    set -e

                    if [[ "$EXIT_CODE" -eq 0 ]]; then
                        ((++PASS))
                        echo "  [PASS] -> $log_file"
                    else
                        ((++FAIL))
                        echo "  [FAIL] Exit=$EXIT_CODE -> $log_file"
                    fi

                    echo "Progress: $COUNT | PASS=$PASS | FAIL=$FAIL"
                    sleep 0.2
                done
            done
        done
    done
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Sweep complete!"
echo "PASS: $PASS / $COUNT"
echo "FAIL: $FAIL / $COUNT"
echo "========================================================"

# ── Auto-generate summary from GLOBAL_RUN_DIR ─────────────────────────
SUMMARY_CSV="$GLOBAL_RUN_DIR/summary_metrics.csv"
if [[ -f "$SUMMARY_CSV" ]]; then
    echo ""
    echo "Generating summary from: $SUMMARY_CSV"
    "$PYTHON" - "$SUMMARY_CSV" "$GLOBAL_RUN_DIR" <<'PYEOF'
import sys, os
import pandas as pd

csv_path = sys.argv[1]
run_dir  = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path)

try:
    df = pd.read_csv(csv_path)
    if 'MAE' in df.columns:
        df_ok = df[df['status'] == 'success'].copy()
        if not df_ok.empty:
            df_ok = df_ok.sort_values('MAE')
            print("\n=== MAE Ranking (Top 10) ===")
            cols = ['model', 'revin_type', 'seq_len', 'pred_len', 'MAE']
            present = [c for c in cols if c in df_ok.columns]
            print(df_ok[present].head(10).to_string(index=False))

            try:
                from plotting import plot_summary_bar
                bar_path = os.path.join(run_dir, 'summary_MAE_bar.png')
                plot_summary_bar(csv_path, metric='MAE', save_path=bar_path)
                print(f"\nBar chart saved: {bar_path}")
            except ImportError:
                pass
except Exception as e:
    print(f"Error during analysis: {e}")

print(f"\nResults in: {run_dir}/")
PYEOF
else
    echo "No summary_metrics.csv found, skipping analysis."
fi

echo ""
echo "All logs: $GLOBAL_RUN_DIR/logs/"
echo "Done."
