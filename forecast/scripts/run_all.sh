#!/bin/bash
# Run all model x dataset x pred_len combinations
# Logs saved to forecast/logs/<setting>.log

cd "$(dirname "$0")/../.."

DATASETS="ETTh1 ETTh2 ETTm1 ETTm2"
PRED_LENS="96 192 336 720"
SEQ_LEN=96
LABEL_LEN=48
FEATURES=M

LOG_DIR="forecast/logs"
mkdir -p "$LOG_DIR"

# Trainable models
for model in DLinear PatchTST; do
    echo "===== $model ====="
    for data in $DATASETS; do
        for pl in $PRED_LENS; do
            setting="${model}_${data}_${FEATURES}_sl${SEQ_LEN}_pl${pl}"
            logfile="${LOG_DIR}/${setting}.log"
            echo ">> $setting -> $logfile"
            python -m forecast.run \
                --model $model --data $data --features $FEATURES \
                --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl \
                --is_training 1 --train_epochs 10 --save_val_pred \
                2>&1 | tee "$logfile"
        done
    done
done

# Zero-shot foundation models
for model in Sundial Chronos Timer TimesFM; do
    echo "===== $model ====="
    for data in $DATASETS; do
        for pl in $PRED_LENS; do
            setting="${model}_${data}_${FEATURES}_sl${SEQ_LEN}_pl${pl}"
            logfile="${LOG_DIR}/${setting}.log"
            echo ">> $setting -> $logfile"
            python -m forecast.run \
                --model $model --data $data --features $FEATURES \
                --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl \
                --is_training 0 --save_val_pred \
                2>&1 | tee "$logfile"
        done
    done
done

# XGBoost Stacking fusion
FUSION_MODELS="DLinear,PatchTST,Sundial,Chronos,Timer,TimesFM"
echo "===== XGBoost Fusion ====="
for data in $DATASETS; do
    for pl in $PRED_LENS; do
        setting="XGBFusion_${data}_${FEATURES}_sl${SEQ_LEN}_pl${pl}"
        logfile="${LOG_DIR}/${setting}.log"
        echo ">> $setting -> $logfile"
        python -m forecast.run \
            --mode fusion --data $data --features $FEATURES \
            --seq_len $SEQ_LEN --pred_len $pl \
            --fusion_models $FUSION_MODELS \
            2>&1 | tee "$logfile"
    done
done

echo "All experiments done!"
echo "Logs saved to $LOG_DIR/"
echo "Run 'python forecast/scripts/parse_logs.py' to generate results CSV"
