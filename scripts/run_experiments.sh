#!/usr/bin/env bash
# =====================================================================
# run_experiments.sh - 时序预测基线系统参数扫描实验脚本
#
# 功能：
#   1. 遍历 seq_len × pred_len × revin_type × top_k 参数网格
#   2. 对每个实验调用 python run.py，单实验结果存入 results/{exp_id}/
#   3. 日志持久化：stdout + stderr 同步写入 results/logs/{exp_id}.log
#   4. 进度条显示（tqdm 进度由 python 端输出）
#   5. 实验结束后汇总 summary_metrics.csv 并打印关键对比
#
# Usage:
#   bash scripts/run_experiments.sh
#   bash scripts/run_experiments.sh --dry-run          # 仅打印命令不执行
#   bash scripts/run_experiments.sh --models PatternSearch,LSHSearch
#   bash scripts/run_experiments.sh --revin-types dual temporal
#
# =====================================================================
set -euo pipefail

# ── 项目根目录 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"

mkdir -p "$LOG_DIR"

# ── 默认参数 ────────────────────────────────────────────────────────
MODELS="${MODELS:-PatternSearch,LSHSearch,SAXSearch}"
SEQ_LENS="${SEQ_LENS:-96,192}"
PRED_LENS="${PRED_LENS:-48,96}"
REVIN_TYPES="${REVIN_TYPES:-none,temporal,feature,dual}"
TOP_K_VALUES="${TOP_K_VALUES:-3,5,10}"
PARALLEL="${PARALLEL:-}"

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --models)       MODELS="$2"; shift 2 ;;
        --revin-types)  REVIN_TYPES="$2"; shift 2 ;;
        --seq-lens)     SEQ_LENS="$2"; shift 2 ;;
        --pred-lens)    PRED_LENS="$2"; shift 2 ;;
        --top-k)        TOP_K_VALUES="$2"; shift 2 ;;
        --parallel)     PARALLEL=1; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra SEQ_ARR <<< "$SEQ_LENS"
IFS=',' read -ra PRED_ARR <<< "$PRED_LENS"
IFS=',' read -ra REVIN_ARR <<< "$REVIN_TYPES"
IFS=',' read -ra TOPK_ARR <<< "$TOP_K_VALUES"

# ── 计算总实验数 ───────────────────────────────────────────────────
TOTAL=0
for model in "${MODEL_ARR[@]}"; do
    for seq in "${SEQ_ARR[@]}"; do
        for pred in "${PRED_ARR[@]}"; do
            for revin in "${REVIN_ARR[@]}"; do
                for k in "${TOPK_ARR[@]}"; do
                    ((TOTAL++)) 2>/dev/null || true
                done
            done
        done
    done
done

echo "========================================================"
echo "时序预测基线系统参数扫描  [v3.0]"
echo "========================================================"
echo "模型:         ${MODEL_ARR[*]}"
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
                    ((COUNT++))

                    # 生成 exp_id（与 run.py _make_exp_id 保持一致）
                    # 格式: dataset_seq{seq}_pred{pred}_k{k}_R{Rev}
                    revin_suffix=""
                    if [[ "$revin" != "none" ]]; then
                        revin_suffix="_R${revin:0:1^}"
                        revin_suffix="_R${revin:0:1}"  # e.g. _Rd (dual), _Rt (temporal)
                        revin_suffix="_R${revin:0:1}"
                    fi

                    # 构造 exp_id（同 run.py _make_exp_id）
                    exp_id="${model}_seq${seq}_pred${pred}_k${k}${revin_suffix}"

                    echo ""
                    echo "[$COUNT/$TOTAL] $exp_id"
                    echo "--------------------------------------------------------------------------"

                    cmd=(
                        "$PYTHON" "$RUN_PY"
                        --model "$model"
                        --seq_len "$seq"
                        --pred_len "$pred"
                        --revin_type "$revin"
                        --top_k "$k"
                    )

                    if [[ -n "$PARALLEL" ]]; then
                        cmd+=(--parallel)
                    fi

                    log_file="$LOG_DIR/${exp_id}.log"

                    if [[ "$DRY_RUN" == true ]]; then
                        echo "DRY RUN: ${cmd[*]}"
                        echo "         log: $log_file"
                        continue
                    fi

                    # ── tee 持久化：stdout + stderr → results/logs/{exp_id}.log ──
                    # 使用 bash process substitution 实现 tee，不污染变量作用域
                    set +e
                    {
                        "${cmd[@]}" 2>&1
                    } | tee "$log_file"
                    EXIT_CODE=${PIPESTATUS[0]}
                    set -e

                    if [[ "$EXIT_CODE" -eq 0 ]]; then
                        ((PASS++))
                        echo "  ✓ [$COUNT/$TOTAL] $exp_id 成功"
                    else
                        ((FAIL++))
                        echo "  ✗ [$COUNT/$TOTAL] $exp_id 失败 (exit $EXIT_CODE)"
                        echo "  查看日志: $log_file"
                    fi

                    echo "--------------------------------------------------------------------------"
                    echo "进度: $COUNT/$TOTAL | 成功: $PASS | 失败: $FAIL"

                    # 小暂停防止文件系统竞争
                    sleep 0.5

                done  # top_k
            done  # revin_type
        done  # pred_len
    done  # seq_len
done  # model

# ── 实验结束 ───────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "实验完成！"
echo "成功: $PASS / $TOTAL"
echo "失败: $FAIL / $TOTAL"
echo "========================================================"

SUMMARY_CSV="$RESULTS_DIR/summary_metrics.csv"
if [[ -f "$SUMMARY_CSV" ]]; then
    echo ""
    echo "Summary CSV 已生成: $SUMMARY_CSV"
    echo ""

    # 打印 MAE 排名
    if command -v python3 &>/dev/null; then
        python3 - <<'PYEOF'
import sys, pandas as pd
csv = sys.argv[1]
try:
    df = pd.read_csv(csv)
    if 'MAE' in df.columns and 'model' in df.columns:
        df_ok = df[df['status'] == 'success'].copy()
        df_ok = df_ok.sort_values('MAE')
        print("\n=== MAE 排名（越低越好）===")
        print("%-20s %-10s %-10s %-8s %-8s" % ("model", "revin", "seq", "pred", "MAE"))
        print("-" * 60)
        for _, row in df_ok.iterrows():
            print("%-20s %-10s %-10s %-8s %.5f"
                % (str(row.get('model',''))[:20],
                   str(row.get('revin_type',''))[:10],
                   str(row.get('seq_len',''))[:10],
                   str(row.get('pred_len',''))[:8],
                   row['MAE']))
        print("")
        # 生成汇总图
        from plotting import plot_summary_bar
        plot_summary_bar(csv, metric='MAE',
                         save_path=csv.replace('.csv', '_MAE_bar.png'),
                         figsize=(max(10, len(df_ok)*1.5), 5))
        plot_summary_bar(csv, metric='MSE',
                         save_path=csv.replace('.csv', '_MSE_bar.png'),
                         figsize=(max(10, len(df_ok)*1.5), 5))
        print("对比柱状图已保存")
except Exception as e:
    print("汇总分析失败:", e)
PYEOF
        "$PYTHON" - "$SUMMARY_CSV" < /dev/null 2>/dev/null \
            || python3 "$SUMMARY_CSV" 2>/dev/null \
            || echo "（Python 汇总分析跳过，请手动查看 CSV）"
    fi
else
    echo "未找到 summary_metrics.csv"
fi

echo ""
echo "所有实验日志保存在: $LOG_DIR/"
echo "========================================================"
