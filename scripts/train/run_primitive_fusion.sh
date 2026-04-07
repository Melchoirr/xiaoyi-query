#!/bin/zsh

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

python -m forecast.run \
  --mode single \
  --model PrimitiveFusion \
  --data ETTh1 \
  --seq_len 512 \
  --pred_len 96 \
  --train_epochs 10 \
  --batch_size 128 \
  --d_model 128 \
  --n_heads 8 \
  --e_layers 3 \
  --d_ff 256 \
  --patch_len 16 \
  --stride 8 \
  --num_primitives 16 \
  --primitive_temp 1.0 \
  --revin \
  --use_gpu \
  2>&1 | tee "$LOG_DIR/PrimitiveFusion_ETTh1_sl512_pl96_temp1.0.log"
