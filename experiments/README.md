# Retrieval Forecasting Experiments

This directory contains the current ETT retrieval experiments and the shared
ETT-style data loader.

## Files

- `data_loader.py`
  - TSL-style dataset/data loader for `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, and
    `custom`.
  - Uses the standard ETT split borders from Time-Series-Library.
  - Exposes `data_provider`, `collect_windows`, and `build_shape_features`.

- `future_factor_retrieval_etth1/channel_independent_direction_retrieval_standalone.py`
  - Current main retrieval experiment.
  - Does full-bank channel-independent retrieval.
  - Baseline score is normalized shape loss only.
  - Rerank score is:

```text
shape_weight * row_norm(shape_loss)
  + direction_weight * row_norm(same_direction_loss)
  + raw_weight * row_norm(raw_loss)
  + std_weight * row_norm(std_loss)
```

- `patchtst_ssl_embedding_retrieval_etth1/`
  - PatchTST/SSL embedding retrieval experiments.
  - Reuses `experiments.data_loader` for ETT windows and shape helpers.

## Method

For each test window, the main experiment retrieves similar train windows from
the complete train bank. Retrieval is channel independent: every channel has its
own candidate scores and top-k set.

The shape loss compares row-normalized history windows. The same-direction loss
compares first-difference signs. A candidate receives a lower direction loss
when more time steps move in the same direction as the query. Each loss matrix
is row-median normalized before weighted combination, so the direction weight is
comparable across samples and channels.

The default final forecast uses inverse-distance averaging over the selected
top-k futures. Before averaging, selected futures are aligned to the query
history statistics with `--align_mode clipped_std`.

## Main Command

Run one ETT setting:

```bash
/Users/zhaodawei/miniconda3/envs/agent/bin/python \
  experiments/future_factor_retrieval_etth1/channel_independent_direction_retrieval_standalone.py \
  --root_path dataset \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --flag test \
  --weights 1.5 \
  --top_k 50 \
  --backend mps \
  --channel_backend batched \
  --selected_batch_size 32 \
  --output_dir outputs/mps_ett_four_datasets_four_lengths/ETTh1_sl96_pl96
```

Important options:

- `--weights`: comma-separated same-direction loss weights.
- `--top_k`: number of retrieved train futures.
- `--backend`: `numpy`, `cpu`, `cuda`, `mps`, or `auto`.
- `--channel_backend loop`: compute each channel separately.
- `--channel_backend batched`: compute all channels in one torch tensor while
  keeping per-channel top-k independent.
- `--selected_batch_size`: query batch size for full-bank scoring. Lower this
  if MPS memory is tight.

## Full ETT Sweep

Run four ETT datasets and four prediction lengths with `seq_len=96`,
`top_k=50`, and direction weight `1.5`:

```bash
bash scripts/experiments/run_mps_ett_four_datasets_four_lengths.sh
```

The script writes summaries to:

```text
outputs/mps_ett_four_datasets_four_lengths/<dataset>_sl96_pl<pred_len>/summary.csv
```

If MPS memory is tight:

```bash
SELECTED_BATCH_SIZE=16 bash scripts/experiments/run_mps_ett_four_datasets_four_lengths.sh
```

To force the older per-channel loop:

```bash
CHANNEL_BACKEND=loop bash scripts/experiments/run_mps_ett_four_datasets_four_lengths.sh
```

## Smoke Checks

Check that the main file compiles:

```bash
/Users/zhaodawei/miniconda3/envs/agent/bin/python -m py_compile \
  experiments/future_factor_retrieval_etth1/channel_independent_direction_retrieval_standalone.py
```

Compare MPS and numpy for ETTh1 seq96 pred96:

```bash
bash scripts/experiments/run_mps_vs_numpy_etth1.sh
```

## Current Best Summary

The compact best-result table is stored at:

```text
outputs/experiment_audit/seq96_best_results.csv
```

Image files are ignored by git. CSV summaries and experiment scripts are kept
for reproducibility.
