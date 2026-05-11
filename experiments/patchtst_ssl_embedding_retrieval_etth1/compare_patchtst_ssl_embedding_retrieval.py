"""
PatchTST self-supervised embedding factor retrieval on ETTh1.

The script pretrains a channel-independent PatchTST encoder with masked-patch
reconstruction on the train split, extracts one embedding per input window, and
uses that embedding as a factor in Top-K retrieval forecasting.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import data_loader as base
from forecast.layers.Embed import PatchEmbedding
from forecast.utils.metrics import metric


FLAGS = ("train", "val", "test")
EPS = 1e-6


def _get_device(preferred="auto"):
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in ("auto", "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred in ("cuda", "mps"):
        print(f"requested device {preferred!r} is unavailable; falling back to CPU")
    return torch.device("cpu")


def _seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = np.ascontiguousarray(windows, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


class PatchTSTMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        seq_len,
        enc_in,
        d_model,
        n_heads,
        e_layers,
        d_ff,
        patch_len,
        stride,
        dropout,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride
        self.patch_num = int((seq_len - patch_len + self.padding) / stride + 1)

        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            padding=self.padding,
            dropout=dropout,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.reconstruction_head = nn.Linear(d_model, patch_len)
        nn.init.normal_(self.mask_token, std=0.02)

    def _patchify(self, x):
        # x: [B, L, C] -> [B*C, P, patch_len]
        bsz, seq_len, channels = x.shape
        x = x.permute(0, 2, 1).reshape(bsz * channels, seq_len)
        x = self.patch_embedding.padding_patch_layer(x.unsqueeze(1)).squeeze(1)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def _position_encoding(self, patch_count, device, dtype):
        return self.patch_embedding.position_encoding.pe[:, :patch_count, :].to(
            device=device,
            dtype=dtype,
        )

    def _tokenize(self, x):
        target_patches = self._patchify(x)
        value_tokens = self.patch_embedding.value_embedding(target_patches)
        pos_tokens = self._position_encoding(
            value_tokens.shape[1],
            value_tokens.device,
            value_tokens.dtype,
        )
        tokens = value_tokens + pos_tokens
        return tokens, pos_tokens, target_patches

    def _encode_patched(self, patch_tokens):
        return self.encoder(patch_tokens)

    def forward(self, x, mask_ratio):
        tokens, pos_tokens, target_patches = self._tokenize(x)

        mask = torch.rand(tokens.shape[:2], device=tokens.device) < mask_ratio
        if not mask.any(dim=1).all():
            empty = ~mask.any(dim=1)
            random_idx = torch.randint(0, tokens.shape[1], (int(empty.sum().item()),), device=tokens.device)
            mask[empty, random_idx] = True

        mask_tokens = self.mask_token.to(tokens.dtype) + pos_tokens
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_tokens, tokens)
        masked_tokens = self.patch_embedding.dropout(masked_tokens)
        encoded = self._encode_patched(masked_tokens)
        reconstructed = self.reconstruction_head(encoded)
        loss = ((reconstructed - target_patches) ** 2)[mask].mean()
        return loss

    @torch.no_grad()
    def encode_window(self, x, pool="mean"):
        bsz, _, channels = x.shape
        tokens, _, _ = self._tokenize(x)
        tokens = self.patch_embedding.dropout(tokens)
        encoded = self._encode_patched(tokens)
        encoded = encoded.reshape(bsz, channels, encoded.shape[1], encoded.shape[2])
        if pool == "mean":
            embedding = encoded.mean(dim=2).reshape(bsz, channels * encoded.shape[-1])
        elif pool == "last":
            embedding = encoded[:, :, -1, :].reshape(bsz, channels * encoded.shape[-1])
        else:
            raise ValueError(f"unsupported embedding pool: {pool}")
        return embedding


def train_ssl_encoder(args, train_windows, val_windows, device, checkpoint_path):
    train_loader = DataLoader(
        WindowDataset(train_windows),
        batch_size=args.ssl_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        WindowDataset(val_windows),
        batch_size=args.ssl_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    model = PatchTSTMaskedAutoencoder(
        seq_len=args.seq_len,
        enc_in=train_windows.shape[2],
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        patch_len=args.patch_len,
        stride=args.stride,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(
        f"SSL pretrain: epochs={args.ssl_epochs}, batch_size={args.ssl_batch_size}, "
        f"mask_ratio={args.mask_ratio}, device={device}"
    )
    if args.ssl_epochs <= 0 and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"--ssl_epochs <= 0 requires an existing checkpoint: {checkpoint_path}"
        )
    for epoch in range(1, args.ssl_epochs + 1):
        started = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, args.mask_ratio)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_ssl_loss(model, val_loader, args.mask_ratio, device)
        train_loss = float(np.mean(train_losses))
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} time={time.time() - started:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded best SSL checkpoint: epoch={state['best_epoch']}, val_loss={state['best_val']:.6f}")
    return model, state


@torch.no_grad()
def evaluate_ssl_loss(model, loader, mask_ratio, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = batch.float().to(device)
        loss = model(batch, mask_ratio)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def extract_embeddings(model, windows, batch_size, device, pool):
    loader = DataLoader(
        WindowDataset(windows),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    embeddings = []
    model.eval()
    for batch in loader:
        batch = batch.float().to(device)
        embeddings.append(model.encode_window(batch, pool=pool).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def fit_embedding_scaler(train_embeddings):
    scaler = StandardScaler()
    scaler.fit(train_embeddings)
    return scaler


def _parse_flags(flags):
    parsed = [flag.strip() for flag in flags.split(",") if flag.strip()]
    invalid = [flag for flag in parsed if flag not in FLAGS]
    if invalid:
        raise ValueError(f"invalid flags: {invalid}; expected any of {FLAGS}")
    if not parsed:
        raise ValueError("at least one flag must be provided")
    return parsed


def _parse_float_list(values):
    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("at least one embedding weight must be provided")
    if any(value < 0 for value in parsed):
        raise ValueError("embedding weights must be non-negative")
    return parsed


def _metric_row(method, flag, setting, metrics, args, ssl_state, extra):
    mae, mse, rmse, mape, mspe = metrics
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "setting": setting,
        "method": method,
        "flag": flag,
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "shape_weight": float(args.shape_weight),
        "embedding_weight": float(extra["embedding_weight"]),
        "top_k": int(args.top_k),
        "batch_size": int(args.retrieval_batch_size),
        "temperature": float(args.temperature),
        "align_futures": int(args.align_futures),
        "align_mode": args.align_mode,
        "ssl_best_epoch": int(ssl_state["best_epoch"]),
        "ssl_best_val": float(ssl_state["best_val"]),
        "embedding_dim": int(extra["embedding_dim"]),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="dataset")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--flags", type=str, default="val,test")
    parser.add_argument("--methods", type=str, default="shape,shape_ssl_embedding")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--retrieval_batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--shape_weight", type=float, default=1.0)
    parser.add_argument("--embedding_weight", type=float, default=0.3)
    parser.add_argument(
        "--embedding_weights",
        type=str,
        default=None,
        help="comma-separated embedding weights to sweep; overrides --embedding_weight",
    )
    parser.add_argument("--align_futures", action="store_true", default=True)
    parser.add_argument("--no_align_futures", action="store_false", dest="align_futures")
    parser.add_argument("--align_mode", type=str, default="std", choices=["std", "mean"])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--ssl_epochs", type=int, default=10)
    parser.add_argument("--ssl_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--embedding_pool", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    _seed_everything(args.seed)
    device = _get_device(args.device)

    data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    setting = (
        f"PatchTSTSSLFactorRetrieval_{data_name}_{args.features}"
        f"_sl{args.seq_len}_pl{args.pred_len}"
    )
    output_dir = args.output_dir or os.path.join(
        "outputs",
        "patchtst_ssl_embedding_retrieval_etth1",
        setting,
    )
    checkpoint_path = args.checkpoint_path or os.path.join(output_dir, "checkpoints", "patchtst_ssl.pth")
    flags = _parse_flags(args.flags)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    embedding_weights = (
        _parse_float_list(args.embedding_weights)
        if args.embedding_weights is not None
        else [args.embedding_weight]
    )
    valid_methods = {"shape", "shape_ssl_embedding"}
    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        raise ValueError(f"invalid methods: {invalid_methods}; expected any of {sorted(valid_methods)}")
    if not methods:
        raise ValueError("at least one method must be provided")

    print(f"PatchTSTSSLFactorRetrieval setting: {setting}")
    print(f"flags={flags}, methods={methods}, embedding_weights={embedding_weights}, output_dir={output_dir}")

    train_set, train_loader = base.data_provider(args, "train")
    val_set, val_loader = base.data_provider(args, "val")
    test_set, test_loader = base.data_provider(args, "test")

    print(f"key columns: {train_set.key_cols}")
    print(f"prediction columns: {train_set.pred_cols}")

    print("building train and validation windows")
    train_windows, train_preds = base.collect_windows(train_loader, args.pred_len)
    val_windows, _ = base.collect_windows(val_loader, args.pred_len)

    ssl_model, ssl_state = train_ssl_encoder(args, train_windows, val_windows, device, checkpoint_path)

    print("extracting train embeddings")
    train_shape = base.build_shape_features(train_windows)
    train_embeddings = extract_embeddings(
        ssl_model,
        train_windows,
        args.retrieval_batch_size,
        device,
        args.embedding_pool,
    )
    embedding_scaler = fit_embedding_scaler(train_embeddings)
    train_embeddings = embedding_scaler.transform(train_embeddings).astype(np.float32)
    print(
        f"train samples={len(train_windows)}, shape_dim={train_shape.shape[1]}, "
        f"embedding_dim={train_embeddings.shape[1]}"
    )

    rows = []
    method_specs = []
    for method in methods:
        if method == "shape":
            method_specs.append(("shape", 0.0))
        else:
            for embedding_weight in embedding_weights:
                method_specs.append((f"{method}_ew{embedding_weight:g}", embedding_weight))

    for method_name, method_embedding_weight in method_specs:
        print(f"\n{'=' * 50}")
        print(f"method={method_name}")

        split_map = {
            "train": (train_windows, train_preds),
            "val": base.collect_windows(val_loader, args.pred_len),
            "test": base.collect_windows(test_loader, args.pred_len),
        }

        for flag in flags:
            print(f"\nprocessing {flag} split")
            target_windows, target_preds = split_map[flag]
            if flag == "train":
                target_shape = train_shape
                target_embeddings = train_embeddings
            else:
                target_shape = base.build_shape_features(target_windows)
                raw_embeddings = extract_embeddings(
                    ssl_model,
                    target_windows,
                    args.retrieval_batch_size,
                    device,
                    args.embedding_pool,
                )
                target_embeddings = embedding_scaler.transform(raw_embeddings).astype(np.float32)

            print(f"{flag} samples: {len(target_windows)}")
            matched_preds, details = base.retrieve_and_predict(
                target_windows=target_windows,
                target_preds=target_preds,
                train_windows=train_windows,
                train_preds=train_preds,
                target_shape=target_shape,
                train_shape=train_shape,
                target_factors=target_embeddings,
                train_factors=train_embeddings,
                shape_weight=args.shape_weight,
                factor_weight=method_embedding_weight,
                top_k=args.top_k,
                batch_size=args.retrieval_batch_size,
                temperature=args.temperature,
                exclude_self=(flag == "train"),
                align_futures=args.align_futures,
                align_mode=args.align_mode,
            )

            mae, mse, rmse, mape, mspe = metric(matched_preds, target_preds)
            print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

            save_dir = os.path.join(output_dir, method_name, flag)
            base._save_split_outputs(
                save_dir,
                matched_preds,
                target_preds,
                details,
                (mae, mse, rmse, mape, mspe),
            )
            print(f"  saved to: {save_dir}")

            rows.append(
                _metric_row(
                    method=method_name,
                    flag=flag,
                    setting=setting,
                    metrics=(mae, mse, rmse, mape, mspe),
                    args=args,
                    ssl_state=ssl_state,
                    extra={
                        "embedding_weight": method_embedding_weight,
                        "embedding_dim": train_embeddings.shape[1],
                    },
                )
            )

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsummary saved to: {summary_path}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        print(
            f"RESULT|{ts}|{row['setting']}|{row['method']}|{row['flag']}|"
            f"mse={row['mse']:.6f}|mae={row['mae']:.6f}|rmse={row['rmse']:.6f}|"
            f"mape={row['mape']:.6f}|mspe={row['mspe']:.6f}"
        )

    print("Done!")


if __name__ == "__main__":
    main()
