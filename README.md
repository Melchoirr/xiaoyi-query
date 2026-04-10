# Retrieval Time Series Forecasting Framework

A unified, auditable framework for long-term time series forecasting via historical window retrieval.

## Task Definition
Given a query history window `X[t-seq_len:t)`, the framework:
1. Retrieves similar history windows `X[s-seq_len:s)` from a **train-only memory bank**.
2. Collects corresponding future windows `Y[s:s+pred_len)`.
3. Aggregates retrieved futures into final prediction `Y_hat[t:t+pred_len)`.

## Supported Models (Retrieval Only)
- `PatternSearch`: exact distance-based retrieval forecaster.
- `LSHSearch`: approximate recall + exact rerank forecaster.
- `SAXSearch`: symbolic coarse recall + exact rerank forecaster.
- `DTWSearch`: elastic-shape (constrained DTW) retrieval forecaster.
- `MatrixProfileSearch`: z-normalized subsequence NN forecaster.
- `TS2VecSearch`: representation retrieval forecaster.
- `RAGSearch`: learned reranking retrieval forecaster.

## Recommended Roles
| Model | Recommended role |
|---|---|
| PatternSearch | Independent forecaster |
| LSHSearch | Recall module + rerank + aggregation |
| SAXSearch | Coarse recall module + rerank + aggregation |
| DTWSearch | Independent forecaster (small/medium memory size) |
| MatrixProfileSearch | Independent forecaster (z-norm subsequence NN) |
| TS2VecSearch | Representation retriever + aggregation |
| RAGSearch | Learned reranker over recalled candidates |

## Normalization Policy
| Model | Policy |
|---|---|
| PatternSearch | `window_zscore` (default) |
| LSHSearch | `window_zscore` (default) |
| SAXSearch | `window_zscore` (default, required by SAX) |
| DTWSearch | `window_zscore` (default) |
| MatrixProfileSearch | `window_zscore` (default) |
| TS2VecSearch | `standard` (train-fit only) |
| RAGSearch | `standard` (train-fit only) |

## Leakage Guards
- Split is chronological (`train -> val -> test`).
- Scaler fit on train split only.
- Memory bank is built from train windows only.
- Query retrieval uses only history windows.
- Future windows are used only as retrieved values and training labels.

## Project Structure
```text
.
©À©¤©¤ run.py
©À©¤©¤ exp/
©¦   ©À©¤©¤ exp_basic.py
©¦   ©À©¤©¤ exp_long_term_forecasting.py
©¦   ©¸©¤©¤ exp_retrieval_forecasting.py
©À©¤©¤ data_provider/
©¦   ©À©¤©¤ data_factory.py
©¦   ©À©¤©¤ data_loader.py
©¦   ©¸©¤©¤ window_builder.py
©À©¤©¤ models/
©¦   ©À©¤©¤ base_retriever.py
©¦   ©À©¤©¤ PatternSearch.py
©¦   ©À©¤©¤ LSHSearch.py
©¦   ©À©¤©¤ SAXSearch.py
©¦   ©À©¤©¤ DTWSearch.py
©¦   ©À©¤©¤ MatrixProfileSearch.py
©¦   ©À©¤©¤ TS2VecSearch.py
©¦   ©¸©¤©¤ RAGSearch.py
©À©¤©¤ retrieval/
©¦   ©À©¤©¤ memory_bank.py
©¦   ©À©¤©¤ candidate_utils.py
©¦   ©À©¤©¤ aggregation.py
©¦   ©À©¤©¤ rerank.py
©¦   ©¸©¤©¤ distance.py
©À©¤©¤ utils/
©¦   ©À©¤©¤ metrics.py
©¦   ©À©¤©¤ seed.py
©¦   ©À©¤©¤ logging_utils.py
©¦   ©¸©¤©¤ audit.py
©À©¤©¤ scripts/
©¦   ©¸©¤©¤ long_term_forecast/
©À©¤©¤ tests/
©¸©¤©¤ README.md
```

## Quick Start
```bash
pip install -r requirements.txt
python run.py --model PatternSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5
```

## Model Examples
```bash
python run.py --model LSHSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5 --recall_k 64
python run.py --model SAXSearch --data ETTh1 --seq_len 96 --pred_len 96 --word_size 8 --alphabet_size 8
python run.py --model DTWSearch --data ETTh1 --seq_len 96 --pred_len 96 --dtw_radius 5
python run.py --model MatrixProfileSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5
python run.py --model TS2VecSearch --data ETTh1 --seq_len 96 --pred_len 96 --ts2vec_epochs 5
python run.py --model RAGSearch --data ETTh1 --seq_len 96 --pred_len 96 --rag_epochs 3
```

## Result Format
`results/summary.csv` contains:
- `model`
- `dataset`
- `seq_len`
- `pred_len`
- `mae`
- `mse`
- `rmse`
- `mape`
- `runtime`

Per-run directory also stores:
- `pred.npy`
- `true.npy`
- `metrics.csv`
- `run.log`

## Testing
```bash
python -m unittest discover -s tests
```

## Notes
- `MAPE/MSPE` use zero-safe denominators and may return `nan` if all targets are exactly zero.
- DTW complexity is high; use smaller memory bank or lower `seq_len` for fast experiments.

## Round-3 Notes (Target-aware Retrieval)

### Target index
- `target_idx` is auto-resolved from dataset columns when `features=M`.
- For ETT datasets with columns `[HUFL,HULL,MUFL,MULL,LUFL,LULL,OT]`, `target=OT` maps to `target_idx=6`.
- Default `-1` is no longer used as implicit target.

### Recommended PatternSearch defaults
- `future_representation=relative_norm`
- `distance_mode=weighted_channel`
- `aggregation_mode=mean` (or `inverse_distance` as close second)
- `rerank_mode=none` for current ETTh1 setting (hybrid rerank did not improve target distance yet)

### Metrics caution
- Prefer MAE/MSE/RMSE/sMAPE for main comparison.
- `MAPE/MSPE` uses configurable `mape_eps` to reduce near-zero instability.
