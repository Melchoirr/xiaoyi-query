"""
可视化 CosineMatch Top-K 逐维度匹配结果。
7维度 × 3样本，每个子图展示：test序列+GT、top-k匹配的历史序列+预测、加权平均预测。
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from forecast.baselines.CosineMatch import (
    load_etth1_data, build_sequences, match_sequences
)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--top_k', type=int, default=5, help='匹配的 K 值')
parser.add_argument('--n_samples', type=int, default=3, help='可视化样本数')
parser.add_argument('--root_path', type=str, default='dataset')
args = parser.parse_args()

SEQ_LEN = args.seq_len
PRED_LEN = args.pred_len
TOP_K = args.top_k
N_SAMPLES = args.n_samples
OUTPUT_DIR = f'forecast/results/CosineMatch_ETTh1_M_sl{SEQ_LEN}_pl{PRED_LEN}'

# ── 加载数据并匹配 ──
print("加载数据...")
splits_norm, splits_raw, col_names, scaler = load_etth1_data(args.root_path, SEQ_LEN, PRED_LEN)
D = len(col_names)

train_seqs, train_preds = build_sequences(splits_norm['train'], SEQ_LEN, PRED_LEN)
train_seqs_raw, _ = build_sequences(splits_raw['train'], SEQ_LEN, PRED_LEN)

test_seqs, test_preds = build_sequences(splits_norm['test'], SEQ_LEN, PRED_LEN)
test_seqs_raw, _ = build_sequences(splits_raw['test'], SEQ_LEN, PRED_LEN)

print("匹配中...")
matched_preds, all_tk_idx, all_tk_dists, all_weights = match_sequences(
    test_seqs, train_seqs, train_preds,
    exclude_self=False, top_k=TOP_K,
    target_seqs_match=test_seqs_raw, train_seqs_match=train_seqs_raw,
    return_details=True)

N_test = len(test_seqs)
sample_indices = np.linspace(0, N_test - 1, N_SAMPLES, dtype=int)

# ── 绘图: 7维 × 3样本 ──
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(D, N_SAMPLES, figsize=(8 * N_SAMPLES, 3.2 * D))
fig.suptitle(f'CosineMatch Top-{TOP_K} 逐维度匹配 (seq={SEQ_LEN}, pred={PRED_LEN})',
             fontsize=16, y=1.005)

x_seq = np.arange(SEQ_LEN)
x_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

# 5 种高饱和度、互相远离的颜色
TOPK_COLORS = ['#FF6600', '#00CC44', '#AA00FF', '#FF0066', '#00AADD']

for col, si in enumerate(sample_indices):
    for row in range(D):
        ax = axes[row, col]

        # test 序列 + ground truth（蓝色，和 top-k 颜色完全不同）
        ax.plot(x_seq, test_seqs[si, :, row], color='#1a1a1a', linewidth=2,
                label='Test 输入', zorder=10)
        ax.plot(x_pred, test_preds[si, :, row], color='#1a1a1a', linewidth=2,
                linestyle='--', label='Test GT', zorder=10)

        # top-k 匹配（每个维度独立的索引）
        tk_idx = all_tk_idx[row, si]      # [K]
        tk_dists = all_tk_dists[row, si]   # [K]
        weights = all_weights[row, si]     # [K]

        for rank in range(TOP_K):
            ki = tk_idx[rank]
            color = TOPK_COLORS[rank]
            lbl = f'#{rank+1} w={weights[rank]:.2f} d={tk_dists[rank]:.1f}'

            ax.plot(x_seq, train_seqs[ki, :, row], color=color, alpha=0.8,
                    linewidth=1.2)
            ax.plot(x_pred, train_preds[ki, :, row], color=color, alpha=0.8,
                    linewidth=1.2, linestyle='--', label=lbl)

        # 加权平均预测（粗红色，最高 zorder）
        ax.plot(x_pred, matched_preds[si, :, row], color='#DC2626', linewidth=3,
                linestyle='-', label='加权预测', zorder=11)

        ax.axvline(x=SEQ_LEN, color='gray', linestyle=':', alpha=0.4)

        # 标注
        if row == 0:
            per_dim_mse = np.mean((matched_preds[si, :, row] - test_preds[si, :, row]) ** 2)
            ax.set_title(f'Sample #{si}', fontsize=11)
        if col == 0:
            ax.set_ylabel(col_names[row], fontsize=10)
        if row == 0 and col == N_SAMPLES - 1:
            ax.legend(fontsize=7, loc='upper right', ncol=2)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'topk_visualization.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存到: {out_path}")
