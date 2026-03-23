# PatternSearch: Memory-based Time Series Forecasting

A minimalist baseline model for time series forecasting based on memory retrieval and k-NN/KD-Tree search, without any deep learning frameworks.

## Overview

This project implements a simple yet effective baseline for time series prediction by:
1. Storing historical input-output pairs as a memory bank
2. Using k-NN (KD-Tree) to find similar patterns
3. Fusing predictions through weighted averaging

## Project Structure

```
xiaoyi-query/
├── data_provider/
│   └── data_loader.py    # ETT data loading, normalization, sliding window
├── models/
│   └── PatternSearch.py  # k-NN based memory retrieval model
├── exp/
│   └── exp_search.py     # Experiment control flow
├── utils/
│   └── metrics.py        # MAE, MSE, RMSE, MAPE, MSPE metrics
├── ETT_data/             # ETT datasets
└── run.py                # Command-line entry point
```

## Installation

```bash
pip install numpy pandas scikit-learn torch
```

## Usage

### Basic Command

```bash
# Single variable prediction on ETTm1
python run.py --data_path ETTm1.csv --features S

# Multi-variable prediction on ETTh1
python run.py --data_path ETTh1.csv --features M
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | ETTm1.csv | Dataset filename |
| `--features` | S | M=multivariate, S=univariate |
| `--seq_len` | 96 | Input sequence length |
| `--pred_len` | 48 | Prediction sequence length |
| `--top_k` | 5 | Number of nearest neighbors |
| `--weighted` | True | Use inverse-distance weighting |
| `--target` | OT | Target column (for univariate mode) |
| `--root_path` | ./ETT_data | Data root directory |
| `--output_dir` | ./results | Output directory |

### Examples

```bash
# Different sequence lengths
python run.py --data_path ETTm1.csv --seq_len 168 --pred_len 96

# Different k values
python run.py --data_path ETTh1.csv --top_k 10

# Simple average (no weighting)
python run.py --data_path ETTm1.csv --no_weighted
```

## Output

Results are saved to `./results/`:
- `*_preds.npy`: Predicted values
- `*_trues.npy`: Ground truth values
- `result.txt`: Evaluation metrics

## Datasets

The project uses ETT (Electricity Transformer Temperature) datasets:
- **ETTh1/ETTh2**: Hourly sampling, 7 features
- **ETTm1/ETTm2**: 15-minute sampling, 7 features

Data split: Train 70% / Val 10% / Test 20%

## Model Details

### PatternSearch Algorithm

```
1. Memory Bank: Store all (X_train, Y_train) pairs
2. Query: For each test sample x, find top-k similar sequences from memory
3. Predict: Fuse corresponding Y values via weighted averaging

Prediction = Σ(w_i * Y_i) / Σ(w_i), where w_i = 1/dist(x, X_i)
```

## License

MIT
