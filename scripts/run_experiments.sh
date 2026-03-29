#!/usr/bin/env bash
# =====================================================================
# run_experiments.sh - 时序预测基线系统参数扫描实验脚本
# =====================================================================
set -euo pipefail

# ── 项目根目录 ──────────────────────────────────────────────────────
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"

mkdir -p "$LOG_DIR"

# ── 默认参数 ────────────────────────────────────────────────────────
MODELS="${MODELS:-PatternSearch,LSHSearch,SAXSearch,DTWSearch,MatrixProfileSearch,TS2VecSearch,RAGSearch}"
SEQ_LENS="${SEQ_LENS:-96}"
PRED_LENS="${PRED_LENS:-96}"
REVIN_TYPES="${REVIN_TYPES:-dual}"
TOP_K_VALUES="${TOP_K_VALUES:-10}"
PARALLEL_FLAG=""

DRY_RUN=false

# ── 参数解析 ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --models)       MODELS="$2"; shift 2 ;;
        --revin-types)  REVIN_TYPES="$2"; shift 2 ;;
        --seq-lens)     SEQ_LENS="$2"; shift 2 ;;
        --pred-lens)    PRED_LENS="$2"; shift 2 ;;
        --top-k)        TOP_K_VALUES="$2"; shift 2 ;;
        --parallel)     PARALLEL_FLAG="--parallel"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 将逗号分隔的字符串转为数组
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra SEQ_ARR <<< "$SEQ_LENS"
IFS=',' read -ra PRED_ARR <<< "$PRED_LENS"
IFS=',' read -ra REVIN_ARR <<< "$REVIN_TYPES"
IFS=',' read -ra TOPK_ARR <<< "$TOP_K_VALUES"

# ── 计算总实验数 ───────────────────────────────────────────────────
TOTAL=$((${#MODEL_ARR[@]} * ${#SEQ_ARR[@]} * ${#PRED_ARR[@]} * ${#REVIN_ARR[@]} * ${#TOPK_ARR[@]}))

echo "========================================================"
echo "时序预测基线系统参数扫描  [v3.0]"
echo "========================================================"
echo "模型:        ${MODEL_ARR[*]}"
echo "seq_len:     ${SEQ_ARR[*]}"
echo "pred_len:    ${PRED_ARR[*]}"
echo "revin_type:  ${REVIN_ARR[*]}"
echo "top_k:       ${TOPK_ARR[*]}"
echo "总计实验数:   $TOTAL"
echo "========================================================"

# ── 主循环 ─────────────────────────────────────────────────────────
RUN_PY="$PROJECT_DIR/run.py"
PYTHON="${PYTHON:-python3}"

COUNT=0
PASS=0
FAIL=0

for model in "${MODEL_ARR[@]}"; do
    for seq in "${SEQ_ARR[@]}"; do
        for pred in "${PRED_ARR[@]}"; do
            for revin in "${REVIN_ARR[@]}"; do
                for k in "${TOPK_ARR[@]}"; do
                    ((++COUNT))

                    # 生成 exp_id 
                    revin_suffix=""
                    if [[ "$revin" != "none" ]]; then
                        # 提取首字母并转小写: dual -> d, temporal -> t
                        char=$(echo "${revin:0:1}" | tr '[:upper:]' '[:lower:]')
                        revin_suffix="_R${char}"
                    fi

                    exp_id="${model}_seq${seq}_pred${pred}_k${k}${revin_suffix}"

                    echo ""
                    echo "[$COUNT/$TOTAL] 正在启动: $exp_id"
                    echo "--------------------------------------------------------------------------"

                    # 构造命令数组
                    cmd=(
                        "$PYTHON" "$RUN_PY"
                        --model "$model"
                        --seq_len "$seq"
                        --pred_len "$pred"
                        --revin_type "$revin"
                        --top_k "$k"
                    )
                    
                    # 增加 parallel 标志（如果用户指定）
                    [[ -z "$PARALLEL_FLAG" ]] || cmd+=("$PARALLEL_FLAG")

                    log_file="$LOG_DIR/${exp_id}.log"

                    if [[ "$DRY_RUN" == true ]]; then
                        echo "DRY RUN: ${cmd[*]}"
                        continue
                    fi

                    # ── 执行并实时显示日志 ──
                    set +e
                    # 使用 PIPESTATUS 捕获管道中第一个命令的退出码
                    "${cmd[@]}" 2>&1 | tee "$log_file"
                    EXIT_CODE=${PIPESTATUS[0]}
                    set -e

                    if [[ "$EXIT_CODE" -eq 0 ]]; then
                        ((++PASS))
                        echo "  ✓ 成功"
                    else
                        ((++FAIL))
                        echo "  ✗ 失败 (Exit Code: $EXIT_CODE)"
                        echo "  日志详情: $log_file"
                    fi

                    echo "进度: $COUNT/$TOTAL | 成功: $PASS | 失败: $FAIL"
                    sleep 0.2
                done
            done
        done
    done
done

# ── 实验汇总分析 ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "实验完成！"
echo "成功: $PASS / $TOTAL"
echo "失败: $FAIL / $TOTAL"
echo "========================================================"

SUMMARY_CSV="$RESULTS_DIR/summary_metrics.csv"
if [[ -f "$SUMMARY_CSV" ]]; then
    echo "正在进行数据汇总分析..."
    # 调用内联 Python 脚本进行分析
    "$PYTHON" - "$SUMMARY_CSV" <<'PYEOF'
import sys, pandas as pd
import os
csv_path = sys.argv[1]
try:
    df = pd.read_csv(csv_path)
    if 'MAE' in df.columns:
        df_ok = df[df['status'] == 'success'].copy()
        if not df_ok.empty:
            df_ok = df_ok.sort_values('MAE')
            print("\n=== MAE 排名（前 10）===")
            cols = ['model', 'revin_type', 'seq_len', 'pred_len', 'MAE']
            present_cols = [c for c in cols if c in df_ok.columns]
            print(df_ok[present_cols].head(10).to_string(index=False))
            
            # 尝试调用绘图逻辑 (如果 plotting.py 存在)
            try:
                from plotting import plot_summary_bar
                plot_summary_bar(csv_path, metric='MAE', save_path=csv_path.replace('.csv', '_MAE.png'))
                print("\n图表已保存至 results/")
            except ImportError:
                print("\n提示: 未找到 plotting.py，跳过可视化。")
except Exception as e:
    print(f"分析过程中出现错误: {e}")
PYEOF
else
    echo "提示: 未检测到 $SUMMARY_CSV，跳过汇总分析。"
fi

echo "所有日志详见: $LOG_DIR/"