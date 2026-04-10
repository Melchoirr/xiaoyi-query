from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.base_retriever import BaseRetrieverForecaster
from retrieval.distance import flatten_windows, l2_distance_matrix


class _PairScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RAGSearch(BaseRetrieverForecaster):
    """Learned reranking retrieval forecaster.

    Pipeline: coarse recall -> learned history-pair scoring -> weighted future aggregation.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
        recall_k: int = 32,
        rag_epochs: int = 3,
        rag_batch_size: int = 64,
        rag_lr: float = 1e-3,
        normalization: str = "standard",
        seed: int = 42,
    ) -> None:
        super().__init__(seq_len, pred_len, top_k, normalization=normalization, aggregation="softmax")
        self.recall_k = max(top_k, recall_k)
        self.rag_epochs = rag_epochs
        self.rag_batch_size = rag_batch_size
        self.rag_lr = rag_lr
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._memory_flat: np.ndarray | None = None
        self._scorer: _PairScorer | None = None

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        hist = self._transform_histories(train_histories)
        self._memory_flat = flatten_windows(hist)

        # Build supervision from train-only oracle future similarity (no val/test leakage).
        future_flat = train_futures.reshape(train_futures.shape[0], -1)
        n = min(2000, self._memory_flat.shape[0])
        idx = np.arange(n)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        idx = idx[: min(512, n)]

        pairs = []
        labels = []
        for i in idx:
            hist_i = self._memory_flat[i]
            fut_err = ((future_flat - future_flat[i:i + 1]) ** 2).mean(axis=1)
            pos = np.argsort(fut_err)[1:4]
            neg = np.argsort(fut_err)[-3:]
            for j in pos:
                feat = np.concatenate([hist_i, self._memory_flat[j], np.abs(hist_i - self._memory_flat[j])], axis=0)
                pairs.append(feat)
                labels.append(1.0)
            for j in neg:
                feat = np.concatenate([hist_i, self._memory_flat[j], np.abs(hist_i - self._memory_flat[j])], axis=0)
                pairs.append(feat)
                labels.append(0.0)

        x = torch.from_numpy(np.asarray(pairs, dtype=np.float32))
        y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=self.rag_batch_size, shuffle=True)

        self._scorer = _PairScorer(in_dim=x.shape[1]).to(self.device)
        opt = torch.optim.Adam(self._scorer.parameters(), lr=self.rag_lr)

        self._scorer.train()
        for _ in range(self.rag_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self._scorer(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def retrieve(self, query_histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = flatten_windows(self._transform_histories(query_histories))
        d = l2_distance_matrix(q, self._memory_flat)
        k = min(self.recall_k, d.shape[1])
        idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(d, idx, axis=1)
        return idx.astype(np.int64), vals.astype(np.float32)

    def rerank(self, query_histories: np.ndarray, candidate_ids: np.ndarray, candidate_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = flatten_windows(self._transform_histories(query_histories))
        b, c = candidate_ids.shape
        final_ids = np.zeros((b, self.top_k), dtype=np.int64)
        final_scores = np.zeros((b, self.top_k), dtype=np.float32)

        self._scorer.eval()
        with torch.no_grad():
            for i in range(b):
                ids = candidate_ids[i]
                cand_hist = self._memory_flat[ids]
                q_rep = np.repeat(q[i:i + 1], repeats=len(ids), axis=0)
                feat = np.concatenate([q_rep, cand_hist, np.abs(q_rep - cand_hist)], axis=1).astype(np.float32)
                logits = self._scorer(torch.from_numpy(feat).to(self.device)).cpu().numpy()
                order = np.argsort(-logits)[: self.top_k]
                chosen = ids[order]
                chosen_logits = logits[order]
                final_ids[i] = chosen
                # aggregation uses distance-like scores, so use negative logits -> smaller is better
                final_scores[i] = -chosen_logits

        return final_ids, final_scores
