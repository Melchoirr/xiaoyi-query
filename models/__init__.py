from __future__ import annotations

from typing import Dict, Type

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
}


def build_model(args) -> BaseRetrieverForecaster:
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY)}")

    common = dict(seq_len=args.seq_len, pred_len=args.pred_len, top_k=args.top_k)

    if args.model == "PatternSearch":
        return PatternSearch(**common, normalization=args.pattern_norm)
    if args.model == "LSHSearch":
        return LSHSearch(
            **common,
            n_hash_funcs=args.n_hash_funcs,
            n_tables=args.n_tables,
            recall_k=args.recall_k,
            normalization=args.lsh_norm,
            random_state=args.seed,
        )
    if args.model == "SAXSearch":
        return SAXSearch(
            **common,
            word_size=args.word_size,
            alphabet_size=args.alphabet_size,
            recall_k=args.recall_k,
            normalization=args.sax_norm,
        )
    if args.model == "DTWSearch":
        return DTWSearch(**common, dtw_radius=args.dtw_radius, normalization=args.dtw_norm)
    if args.model == "MatrixProfileSearch":
        return MatrixProfileSearch(**common, normalization=args.mp_norm)
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

    raise RuntimeError("unreachable")
