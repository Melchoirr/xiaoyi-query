#!/bin/bash
# Run all model x dataset x pred_len combinations

cd "$(dirname "$0")/../.."

DATASETS="ETTh1 ETTh2 ETTm1 ETTm2"
PRED_LENS="96 192 336 720"
SEQ_LEN=96
LABEL_LEN=48

echo "===== DLinear ====="
for data in $DATASETS; do
    for pl in $PRED_LENS; do
        echo ">> DLinear | $data | pred_len=$pl"
        python -m forecast.run --model DLinear --data $data --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --is_training 1 --train_epochs 10
    done
done

echo "===== PatchTST ====="
for data in $DATASETS; do
    for pl in $PRED_LENS; do
        echo ">> PatchTST | $data | pred_len=$pl"
        python -m forecast.run --model PatchTST --data $data --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --is_training 1 --train_epochs 10
    done
done

echo "===== Sundial ====="
for data in $DATASETS; do
    for pl in $PRED_LENS; do
        echo ">> Sundial | $data | pred_len=$pl"
        python -m forecast.run --model Sundial --data $data --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --is_training 0
    done
done

echo "All experiments done!"
