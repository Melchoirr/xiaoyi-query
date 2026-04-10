from __future__ import annotations

from typing import Dict, Type

from models.Baselines import HistoricalMean, RepeatLastValue, Top1NearestFuture
from models.DTWSearch import DTWSearch
from models.LSHSearch import LSHSearch
from models.MatrixProfileSearch import MatrixProfileSearch
from models.PatternSearch import PatternSearch
from models.RAGSearch import RAGSearch
from models.SAXSearch import SAXSearch
from models.TS2VecSearch import TS2VecSearch
from models.base_retriever import BaseRetrieverForecaster


MODEL_REGISTRY: Dict[str, Type[BaseRetrieverForecaster]] = {
    "PatternSearch": PatternSearch,
    "LSHSearch": LSHSearch,
    "SAXSearch": SAXSearch,
    "DTWSearch": DTWSearch,
    "MatrixProfileSearch": MatrixProfileSearch,
    "TS2VecSearch": TS2VecSearch,
    "RAGSearch": RAGSearch,
    "RepeatLastValue": RepeatLastValue,
    "HistoricalMean": HistoricalMean,
    "Top1NearestFuture": Top1NearestFuture,
}


def parse_channel_weights(s: str | None):
    if s is None or s.strip() == "":
        return None
    return [float(x) for x in s.split(",")]


def build_model(args) -> BaseRetrieverForecaster:
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY)}")

    common = dict(seq_len=args.seq_len, pred_len=args.pred_len, top_k=args.top_k)
    common_retr = dict(
        aggregation_mode=args.aggregation_mode,
        aggregation_temperature=args.aggregation_temperature,
        future_representation=args.future_representation,
        restoration_mode=args.restoration_mode,
        distance_mode=args.distance_mode,
        channel_weights=parse_channel_weights(args.channel_weights),
        target_idx=args.target_idx,
        rerank_mode=args.rerank_mode,
        rerank_alpha=args.rerank_alpha,
        rerank_beta=args.rerank_beta,
        rerank_gamma=args.rerank_gamma,
        rerank_delta=args.rerank_delta,
    )

    if args.model == "PatternSearch":
        return PatternSearch(**common, normalization=args.pattern_norm, **common_retr)
    if args.model == "LSHSearch":
        return LSHSearch(
            **common,
            n_hash_funcs=args.n_hash_funcs,
            n_tables=args.n_tables,
            recall_k=args.recall_k,
            normalization=args.lsh_norm,
            random_state=args.seed,
            **common_retr,
        )
    if args.model == "SAXSearch":
        return SAXSearch(
            **common,
            word_size=args.word_size,
            alphabet_size=args.alphabet_size,
            recall_k=args.recall_k,
            normalization=args.sax_norm,
            **common_retr,
        )
    if args.model == "DTWSearch":
        return DTWSearch(**common, dtw_radius=args.dtw_radius, normalization=args.dtw_norm, **common_retr)
    if args.model == "MatrixProfileSearch":
        return MatrixProfileSearch(
            **common,
            normalization=args.mp_norm,
            aggregation_mode=args.aggregation_mode,
            future_representation=args.future_representation,
            restoration_mode=args.restoration_mode,
        )
    if args.model == "TS2VecSearch":
        return TS2VecSearch(
            **common,
            hidden_dim=args.hidden_dim,
            epochs=args.ts2vec_epochs,
            batch_size=args.ts2vec_batch_size,
            lr=args.ts2vec_lr,
            normalization=args.ts2vec_norm,
            seed=args.seed,
        )
    if args.model == "RAGSearch":
        return RAGSearch(
            **common,
            recall_k=args.recall_k,
            rag_epochs=args.rag_epochs,
            rag_batch_size=args.rag_batch_size,
            rag_lr=args.rag_lr,
            normalization=args.rag_norm,
            seed=args.seed,
        )
    if args.model == "RepeatLastValue":
        return RepeatLastValue(seq_len=args.seq_len, pred_len=args.pred_len)
    if args.model == "HistoricalMean":
        return HistoricalMean(seq_len=args.seq_len, pred_len=args.pred_len)
    if args.model == "Top1NearestFuture":
        return Top1NearestFuture(seq_len=args.seq_len, pred_len=args.pred_len)

    raise RuntimeError("unreachable")
