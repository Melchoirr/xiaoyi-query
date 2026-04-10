from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.base_retriever import BaseRetrieverForecaster
from retrieval.candidate_utils import topk_from_distances
from retrieval.distance import cosine_similarity_matrix


class _TS2VecEncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        z = self.net(x.transpose(1, 2))
        return z.mean(dim=-1)


class TS2VecSearch(BaseRetrieverForecaster):
    """Representation retrieval forecaster (train encoder on train split only)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int = 5,
        hidden_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
        normalization: str = "standard",
        seed: int = 42,
    ) -> None:
        super().__init__(seq_len, pred_len, top_k, normalization=normalization, aggregation="softmax")
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder: _TS2VecEncoder | None = None
        self.memory_embeddings: np.ndarray | None = None

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.05
        return x + noise

    def _fit_model(self, train_histories: np.ndarray, train_futures: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        x = self._transform_histories(train_histories)
        dataset = TensorDataset(torch.from_numpy(x).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.encoder = _TS2VecEncoder(channels=x.shape[-1], hidden_dim=self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

        self.encoder.train()
        for _ in range(self.epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                v1 = self._augment(xb)
                v2 = self._augment(xb)
                z1 = F.normalize(self.encoder(v1), dim=-1)
                z2 = F.normalize(self.encoder(v2), dim=-1)
                logits = z1 @ z2.T
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.encoder.eval()
        with torch.no_grad():
            emb = self.encoder(torch.from_numpy(x).float().to(self.device)).cpu().numpy().astype(np.float32)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        self.memory_embeddings = emb

    def retrieve(self, query_histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._transform_histories(query_histories)
        with torch.no_grad():
            q = self.encoder(torch.from_numpy(x).float().to(self.device)).cpu().numpy().astype(np.float32)
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
        sim = cosine_similarity_matrix(q, self.memory_embeddings)
        dist = 1.0 - sim
        return topk_from_distances(dist.astype(np.float32), self.top_k)
