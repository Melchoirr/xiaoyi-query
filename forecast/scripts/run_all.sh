#!/bin/bash
# Run all model x dataset x pred_len combinations
# Logs saved to forecast/logs/<setting>.log

cd "$(dirname "$0")/../.."

DATASETS="ETTh1"
PRED_LENS="96"
SEQ_LEN=512
FEATURES=M

LOG_DIR="forecast/logs"
mkdir -p "$LOG_DIR"

# DLinear (original paper: lr=0.005, batch=32, epochs=10)
# for data in $DATASETS; do
#     for pl in $PRED_LENS; do
#         setting="DLinear_${data}_${FEATURES}_sl${SEQ_LEN}_pl${pl}"
#         logfile="${LOG_DIR}/${setting}.log"
#         echo ">> 启动 $setting -> $logfile"
#         python -u -m forecast.run \
#             --model DLinear --data $data --features $FEATURES \
#             --seq_len $SEQ_LEN --label_len 48 --pred_len $pl \
#             --is_training 1 --train_epochs 10 --patience 3 \
#             --learning_rate 0.005 --batch_size 32 \
#             --use_gpu \
#             --save_val_pred --save_train_pred \
#             2>&1 | tee "$logfile"
#     done
# done

# # PatchTST (original paper: lr=0.0001, batch=128, epochs=100,
# #            d_model=16, n_heads=4, e_layers=3, d_ff=128,
# #            fc_dropout=0.3, head_dropout=0, revin=1, lradj=type3, patience=100)
# for data in $DATASETS; do
#     for pl in $PRED_LENS; do
#         setting="PatchTST_${data}_${FEATURES}_sl${SEQ_LEN}_pl${pl}"
#         logfile="${LOG_DIR}/${setting}.log"
#         echo ">> 启动 $setting -> $logfile"
#         python -u -m forecast.run \
#             --model PatchTST --data $data --features $FEATURES \
#             --seq_len $SEQ_LEN --label_len 48 --pred_len $pl \
#             --is_training 1 --train_epochs 100 --patience 100 \
#             --learning_rate 0.0001 --batch_size 128 --lradj type3 \
#             --d_model 32 --n_heads 4 --e_layers 3 --d_ff 256 \
#             --patch_len 16 --stride 8 \
#             --dropout 0.2 --fc_dropout 0 --head_dropout 0 \
#             --revin --use_gpu \
#             --save_val_pred --save_train_pred \
#             2>&1 | tee "$logfile"
#     done
# done

# Zero-shot foundation models (串行，避免显存冲突)
# 使用 benchmark.py 统一入口，支持 train/val/test 三种 flag
SCRIPT_DIR="$(dirname "$0")"
for model in chronos timerxl sundial moirai; do
    for data in $DATASETS; do
        for pl in $PRED_LENS; do
            logfile="${LOG_DIR}/${model}_${data}_${FEATURES}_sl${SEQ_LEN}_pl${pl}.log"
            echo ">> 启动 ${model} ${data} pl${pl} -> $logfile"
            python -u "${SCRIPT_DIR}/benchmark.py" \
                --model $model --dataset $data --pred_len $pl \
                --seq_len $SEQ_LEN --features $FEATURES \
                --flags test,val,train \
                2>&1 | tee "$logfile"
        done
    done
done

# XGBoost Stacking fusion
FUSION_MODELS="DLinear,PatchTST,Chronos,Timer,Sundial,Moirai"
for data in $DATASETS; do
    for pl in $PRED_LENS; do
        setting="XGBFusion_${data}_${FEATURES}_sl${SEQ_LEN}_pl${pl}"
        logfile="${LOG_DIR}/${setting}.log"
        echo ">> 启动 $setting -> $logfile"
        python -u -m forecast.run \
            --mode fusion --data $data --features $FEATURES \
            --seq_len $SEQ_LEN --pred_len $pl \
            --fusion_models $FUSION_MODELS \
            2>&1 | tee "$logfile"
    done
done

echo "All experiments done! Logs saved to $LOG_DIR/"
