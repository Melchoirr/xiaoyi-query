import argparse
import csv
import os
import pandas as pd

from exp.exp_retrieval_forecasting import Exp_Retrieval_Forecasting
from utils.logging_utils import configure_logger
from utils.seed import set_global_seed


ALL_MODELS = [
    "PatternSearch", "LSHSearch", "SAXSearch", "DTWSearch", "MatrixProfileSearch", "TS2VecSearch", "RAGSearch",
    "RepeatLastValue", "HistoricalMean", "Top1NearestFuture",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retrieval-based time series forecasting")

    p.add_argument("--model", type=str, default="PatternSearch", choices=ALL_MODELS)
    p.add_argument("--data", type=str, default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    p.add_argument("--root_path", type=str, default="./dataset/")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    p.add_argument("--target", type=str, default="OT")
    p.add_argument("--freq", type=str, default="h")

    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--recall_k", type=int, default=64)

    p.add_argument("--n_hash_funcs", type=int, default=16)
    p.add_argument("--n_tables", type=int, default=4)
    p.add_argument("--word_size", type=int, default=8)
    p.add_argument("--alphabet_size", type=int, default=8)
    p.add_argument("--dtw_radius", type=int, default=5)

    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--ts2vec_epochs", type=int, default=5)
    p.add_argument("--ts2vec_batch_size", type=int, default=64)
    p.add_argument("--ts2vec_lr", type=float, default=1e-3)

    p.add_argument("--rag_epochs", type=int, default=3)
    p.add_argument("--rag_batch_size", type=int, default=64)
    p.add_argument("--rag_lr", type=float, default=1e-3)

    p.add_argument("--pattern_norm", type=str, default="window_zscore")
    p.add_argument("--lsh_norm", type=str, default="window_zscore")
    p.add_argument("--sax_norm", type=str, default="window_zscore")
    p.add_argument("--dtw_norm", type=str, default="window_zscore")
    p.add_argument("--mp_norm", type=str, default="window_zscore")
    p.add_argument("--ts2vec_norm", type=str, default="standard")
    p.add_argument("--rag_norm", type=str, default="standard")

    p.add_argument("--future_representation", type=str, default="relative_norm", choices=["raw", "relative_norm", "delta"])
    p.add_argument("--restoration_mode", type=str, default="auto", choices=["auto", "raw", "none", "relative_norm", "history_stat_norm", "delta"])
    p.add_argument("--distance_mode", type=str, default="weighted_channel", choices=["target_only", "all_channel_flat", "weighted_channel", "summary_augmented"])
    p.add_argument("--channel_weights", type=str, default="")
    p.add_argument("--target_idx", type=int, default=None)

    p.add_argument("--rerank_mode", type=str, default="none", choices=["none", "exact", "hybrid"])
    p.add_argument("--rerank_alpha", type=float, default=1.0)
    p.add_argument("--rerank_beta", type=float, default=0.5)
    p.add_argument("--rerank_gamma", type=float, default=0.3)
    p.add_argument("--rerank_delta", type=float, default=0.2)

    p.add_argument("--aggregation_mode", type=str, default="inverse_distance", choices=["mean", "inverse_distance", "softmax_temp", "rank_based", "top1"])
    p.add_argument("--aggregation_temperature", type=float, default=1.0)

    p.add_argument("--mape_eps", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoints", type=str, default="./checkpoints/")
    p.add_argument("--result_path", type=str, default="./results/")
    p.add_argument("--eval_on_original_scale", type=lambda x: x.lower() == "true", default=True)
    return p


def validate_args(args) -> None:
    if args.seq_len <= 0 or args.pred_len <= 0:
        raise ValueError("seq_len and pred_len must be > 0")
    if args.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if args.recall_k <= 0:
        raise ValueError("recall_k must be > 0")
    if args.target_idx is not None and args.target_idx < 0:
        raise ValueError("target_idx must be >=0 when specified")


def infer_dataset_defaults(args) -> None:
    mapping = {
        "ETTh1": ("ETTh1.csv", "h"),
        "ETTh2": ("ETTh2.csv", "h"),
        "ETTm1": ("ETTm1.csv", "t"),
        "ETTm2": ("ETTm2.csv", "t"),
    }
    default_path, default_freq = mapping[args.data]
    if args.data_path == "ETTh1.csv" and args.data != "ETTh1":
        args.data_path = default_path
    if args.freq == "h" and default_freq != "h":
        args.freq = default_freq


def resolve_target_idx(args) -> None:
    if args.target_idx is not None:
        return
    if args.features == "S":
        args.target_idx = 0
        return
    csv_path = os.path.join(args.root_path, args.data_path)
    cols = list(pd.read_csv(csv_path, nrows=1).columns)
    feature_cols = cols[1:]
    if args.target not in feature_cols:
        raise ValueError(f"target `{args.target}` not found in feature columns: {feature_cols}")
    args.target_idx = feature_cols.index(args.target)


def append_summary_csv(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = ["model", "dataset", "seq_len", "pred_len", "mae", "mse", "rmse", "mape", "smape", "runtime"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    infer_dataset_defaults(args)
    resolve_target_idx(args)

    setting = f"{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_k{args.top_k}_fut{args.future_representation}_dist{args.distance_mode}_agg{args.aggregation_mode}_rer{args.rerank_mode}"
    log_dir = os.path.join(args.result_path, setting)
    logger = configure_logger(log_dir)

    set_global_seed(args.seed, deterministic=True)
    logger.info("Args: %s", vars(args))
    logger.info(
        "Effective target config | target_name=%s target_idx=%s distance_mode=%s channel_weights=%s",
        args.target,
        args.target_idx,
        args.distance_mode,
        args.channel_weights,
    )

    exp = Exp_Retrieval_Forecasting(args)
    exp.train(setting)
    metrics = exp.test(setting)

    summary_row = {
        "model": args.model,
        "dataset": args.data,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "smape": metrics.get("smape", ""),
        "runtime": metrics["runtime"],
    }
    append_summary_csv(os.path.join(args.result_path, "summary.csv"), summary_row)
    logger.info("Test metrics: %s", metrics)


if __name__ == "__main__":
    main()
