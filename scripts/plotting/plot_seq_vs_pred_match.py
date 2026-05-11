"""
对比可视化：SeqMatch（用历史序列匹配）vs PredMatch（用未来值匹配，oracle上界）。
每个样本一行两列，左=SeqMatch，右=PredMatch，7维度叠在同一子图中分行。
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from forecast.baselines.CosineMatch import (
    load_etth1_data, build_sequences,
    match_sequences, _get_device
)
from forecast.baselines.PredMatch import match_by_pred

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--n_samples', type=int, default=3)
parser.add_argument('--root_path', type=str, default='dataset')
args = parser.parse_args()

SEQ_LEN = args.seq_len
PRED_LEN = args.pred_len
TOP_K = args.top_k
N_SAMPLES = args.n_samples
OUTPUT_DIR = f'outputs/results/CosineMatch_ETTh1_M_sl{SEQ_LEN}_pl{PRED_LEN}'

# ── 加载数据 ──
print("加载数据...")
splits_norm, splits_raw, col_names, scaler = load_etth1_data(args.root_path, SEQ_LEN, PRED_LEN)
D = len(col_names)

train_seqs, train_preds = build_sequences(splits_norm['train'], SEQ_LEN, PRED_LEN)
train_seqs_raw, _ = build_sequences(splits_raw['train'], SEQ_LEN, PRED_LEN)
test_seqs, test_preds = build_sequences(splits_norm['test'], SEQ_LEN, PRED_LEN)
test_seqs_raw, _ = build_sequences(splits_raw['test'], SEQ_LEN, PRED_LEN)

# ── SeqMatch: 用历史序列匹配（return_details） ──
print("SeqMatch 匹配中...")
seq_matched, seq_tk_idx, seq_tk_dists, seq_tk_weights = match_sequences(
    test_seqs, train_seqs, train_preds,
    exclude_self=False, top_k=TOP_K,
    target_seqs_match=test_seqs_raw, train_seqs_match=train_seqs_raw,
    return_details=True)

# ── PredMatch: 用未来值匹配（需要手动提取 top-k 详情） ──
print("PredMatch 匹配中...")
import torch

device = _get_device()
N_test = len(test_preds)
pred_tk_idx = np.zeros((D, N_test, TOP_K), dtype=int)
pred_tk_dists = np.zeros((D, N_test, TOP_K))
pred_tk_weights = np.zeros((D, N_test, TOP_K))
pred_matched = np.zeros_like(test_preds)

for d in range(D):
    T = torch.from_numpy(train_preds[:, :, d]).float().to(device)
    T_sq = (T ** 2).sum(dim=1)

    for start in range(0, N_test, 512):
        end = min(start + 512, N_test)
        Q = torch.from_numpy(test_preds[start:end, :, d]).float().to(device)
        B = Q.shape[0]

        Q_sq = (Q ** 2).sum(dim=1, keepdim=True)
        dists = Q_sq + T_sq.unsqueeze(0) - 2.0 * (Q @ T.T)

        tk_d, tk_i = torch.topk(dists, TOP_K, dim=1, largest=False)
        w = 1.0 / (tk_d + 1e-8)
        w = w / w.sum(dim=1, keepdim=True)

        gathered = T[tk_i.reshape(-1)].reshape(B, TOP_K, PRED_LEN)
        weighted = (w.unsqueeze(-1) * gathered).sum(dim=1)

        pred_tk_idx[d, start:end] = tk_i.cpu().numpy()
        pred_tk_dists[d, start:end] = tk_d.cpu().numpy()
        pred_tk_weights[d, start:end] = w.cpu().numpy()
        pred_matched[start:end, :, d] = weighted.cpu().numpy()

from forecast.utils.metrics import metric
seq_mae, seq_mse, *_ = metric(seq_matched, test_preds)
pred_mae, pred_mse, *_ = metric(pred_matched, test_preds)
print(f"SeqMatch  MSE={seq_mse:.6f} MAE={seq_mae:.6f}")
print(f"PredMatch MSE={pred_mse:.6f} MAE={pred_mae:.6f}")

# ── 绘图: 7维 × 3样本 × 2方法 ──
sample_indices = np.linspace(0, N_test - 1, N_SAMPLES, dtype=int)

TOPK_COLORS = ['#FF6600', '#00CC44', '#AA00FF', '#FF0066', '#00AADD']
x_seq = np.arange(SEQ_LEN)
x_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(D, N_SAMPLES * 2, figsize=(7 * N_SAMPLES * 2, 3 * D))
fig.suptitle(
    f'SeqMatch vs PredMatch (oracle)  |  Top-K={TOP_K}  seq={SEQ_LEN} pred={PRED_LEN}\n'
    f'SeqMatch MSE={seq_mse:.4f}  |  PredMatch MSE={pred_mse:.4f}',
    fontsize=15, y=1.005)

for sample_col, si in enumerate(sample_indices):
    for row in range(D):
        # ── 左列: SeqMatch ──
        ax_seq = axes[row, sample_col * 2]

        ax_seq.plot(x_seq, test_seqs[si, :, row], color='#1a1a1a', linewidth=2,
                    zorder=10)
        ax_seq.plot(x_pred, test_preds[si, :, row], color='#1a1a1a', linewidth=2,
                    linestyle='--', zorder=10, label='GT')

        for rank in range(TOP_K):
            ki = seq_tk_idx[row, si, rank]
            w = seq_tk_weights[row, si, rank]
            color = TOPK_COLORS[rank]
            ax_seq.plot(x_seq, train_seqs[ki, :, row], color=color, alpha=0.7, linewidth=1)
            ax_seq.plot(x_pred, train_preds[ki, :, row], color=color, alpha=0.7,
                        linewidth=1, linestyle='--',
                        label=f'#{rank+1} w={w:.2f}')

        ax_seq.plot(x_pred, seq_matched[si, :, row], color='#DC2626', linewidth=2.5,
                    zorder=11, label='加权预测')
        ax_seq.axvline(x=SEQ_LEN, color='gray', linestyle=':', alpha=0.4)

        if row == 0:
            ax_seq.set_title(f'Sample #{si} — SeqMatch', fontsize=10)
        if sample_col == 0:
            ax_seq.set_ylabel(col_names[row], fontsize=9)
        if row == 0 and sample_col == N_SAMPLES - 1:
            ax_seq.legend(fontsize=6, loc='upper right', ncol=2)

        # ── 右列: PredMatch ──
        ax_pred = axes[row, sample_col * 2 + 1]

        ax_pred.plot(x_seq, test_seqs[si, :, row], color='#1a1a1a', linewidth=2,
                     zorder=10)
        ax_pred.plot(x_pred, test_preds[si, :, row], color='#1a1a1a', linewidth=2,
                     linestyle='--', zorder=10, label='GT')

        for rank in range(TOP_K):
            ki = pred_tk_idx[row, si, rank]
            w = pred_tk_weights[row, si, rank]
            color = TOPK_COLORS[rank]
            ax_pred.plot(x_seq, train_seqs[ki, :, row], color=color, alpha=0.7, linewidth=1)
            ax_pred.plot(x_pred, train_preds[ki, :, row], color=color, alpha=0.7,
                         linewidth=1, linestyle='--',
                         label=f'#{rank+1} w={w:.2f}')

        ax_pred.plot(x_pred, pred_matched[si, :, row], color='#DC2626', linewidth=2.5,
                     zorder=11, label='加权预测')
        ax_pred.axvline(x=SEQ_LEN, color='gray', linestyle=':', alpha=0.4)

        if row == 0:
            ax_pred.set_title(f'Sample #{si} — PredMatch (oracle)', fontsize=10)
        if row == 0 and sample_col == N_SAMPLES - 1:
            ax_pred.legend(fontsize=6, loc='upper right', ncol=2)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'seq_vs_pred_match.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存到: {out_path}")
