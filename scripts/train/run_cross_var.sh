#!/bin/bash
# 跨变量预测实验: 用变量A的seq训练DLinear预测变量B的pred
# 7个变量 x 7个变量 = 49次训练

cd "$(dirname "$0")/../.."

DATA=ETTh1
SEQ_LEN=192
PRED_LEN=96
N_CHANNELS=7  # ETTh1: HUFL(0) HULL(1) MUFL(2) MULL(3) LUFL(4) LULL(5) OT(6)

LOG_DIR="forecast/logs/cross_var"
mkdir -p "$LOG_DIR"

for src in $(seq 0 $((N_CHANNELS - 1))); do
    for tgt in $(seq 0 $((N_CHANNELS - 1))); do
        setting="DLinear_${DATA}_crossvar_src${src}_tgt${tgt}_sl${SEQ_LEN}_pl${PRED_LEN}"
        logfile="${LOG_DIR}/${setting}.log"
        echo ">> [src=${src} -> tgt=${tgt}] $setting"
        python -u -m forecast.run \
            --mode cross_var \
            --model DLinear --data $DATA \
            --seq_len $SEQ_LEN --label_len 48 --pred_len $PRED_LEN \
            --src_channel $src --tgt_channel $tgt \
            --is_training 1 --train_epochs 15 --patience 5 \
            --learning_rate 0.005 --batch_size 32 \
            --use_gpu \
            --save_val_pred --save_train_pred \
            2>&1 | tee "$logfile"
    done
done

echo "All 49 cross-var experiments done! Logs in $LOG_DIR/"
