"""
分析 SeqMatch vs PredMatch 匹配到的训练序列，在哪些统计维度上
PredMatch 的 train seq 与 test seq 更接近。
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from forecast.baselines.CosineMatch import (
    load_etth1_data, build_sequences, match_sequences, _get_device
)

# ── 统计特征提取 ──
def extract_stats(seqs):
    """对 [N, seq_len, D] 序列提取多种统计特征，返回 dict of [N, D]"""
    N, L, D = seqs.shape
    stats = {}

    stats['mean'] = seqs.mean(axis=1)                          # 均值
    stats['std'] = seqs.std(axis=1)                            # 标准差
    stats['last_val'] = seqs[:, -1, :]                         # 最后一个值
    stats['first_val'] = seqs[:, 0, :]                         # 第一个值

    # 趋势斜率 (线性回归斜率)
    x = np.arange(L, dtype=float)
    x_centered = x - x.mean()
    slopes = np.zeros((N, D))
    for d in range(D):
        # slope = Σ(x_c * y) / Σ(x_c^2)
        y = seqs[:, :, d]  # [N, L]
        slopes[:, d] = (y * x_centered).sum(axis=1) / (x_centered ** 2).sum()
    stats['slope'] = slopes

    # 末段趋势 (最后 1/4 的斜率)
    quarter = L // 4
    x_q = np.arange(quarter, dtype=float)
    x_q_c = x_q - x_q.mean()
    tail_slopes = np.zeros((N, D))
    for d in range(D):
        y = seqs[:, -quarter:, d]
        tail_slopes[:, d] = (y * x_q_c).sum(axis=1) / (x_q_c ** 2).sum()
    stats['tail_slope'] = tail_slopes

    # 变化率 (最后一个值 - 第一个值)
    stats['range'] = seqs[:, -1, :] - seqs[:, 0, :]

    # 末段均值 (最后 1/4)
    stats['tail_mean'] = seqs[:, -quarter:, :].mean(axis=1)

    # 末段标准差
    stats['tail_std'] = seqs[:, -quarter:, :].std(axis=1)

    # 一阶差分的均值和标准差 (捕捉动量和波动性)
    diff = np.diff(seqs, axis=1)  # [N, L-1, D]
    stats['diff_mean'] = diff.mean(axis=1)
    stats['diff_std'] = diff.std(axis=1)

    # 末段一阶差分均值 (末段动量)
    stats['tail_diff_mean'] = diff[:, -quarter:, :].mean(axis=1)

    # 自相关 (lag=1)
    y = seqs - seqs.mean(axis=1, keepdims=True)
    autocorr = np.zeros((N, D))
    for d in range(D):
        y_d = y[:, :, d]
        var = (y_d ** 2).sum(axis=1) + 1e-8
        autocorr[:, d] = (y_d[:, :-1] * y_d[:, 1:]).sum(axis=1) / var
    stats['autocorr_lag1'] = autocorr

    return stats


# ── 主流程 ──
print("加载数据...")
splits_norm, splits_raw, col_names, scaler = load_etth1_data('dataset', 96, 96)
D = len(col_names)

train_seqs, train_preds = build_sequences(splits_norm['train'], 96, 96)
train_seqs_raw, _ = build_sequences(splits_raw['train'], 96, 96)
test_seqs, test_preds = build_sequences(splits_norm['test'], 96, 96)
test_seqs_raw, _ = build_sequences(splits_raw['test'], 96, 96)
N_test = len(test_seqs)

# ── SeqMatch top-1 索引 ──
print("SeqMatch...")
_, seq_tk_idx, _, _ = match_sequences(
    test_seqs, train_seqs, train_preds,
    exclude_self=False, top_k=1,
    target_seqs_match=test_seqs_raw, train_seqs_match=train_seqs_raw,
    return_details=True)
# seq_tk_idx: [D, N_test, 1]

# ── PredMatch top-1 索引 ──
print("PredMatch...")
device = _get_device()
pred_tk_idx = np.zeros((D, N_test, 1), dtype=int)
for d in range(D):
    T = torch.from_numpy(train_preds[:, :, d]).float().to(device)
    T_sq = (T ** 2).sum(dim=1)
    for start in range(0, N_test, 512):
        end = min(start + 512, N_test)
        Q = torch.from_numpy(test_preds[start:end, :, d]).float().to(device)
        Q_sq = (Q ** 2).sum(dim=1, keepdim=True)
        dists = Q_sq + T_sq.unsqueeze(0) - 2.0 * (Q @ T.T)
        _, tk_i = torch.topk(dists, 1, dim=1, largest=False)
        pred_tk_idx[d, start:end] = tk_i.cpu().numpy()

# ── 提取统计特征 ──
print("提取统计特征...")
test_stats = extract_stats(test_seqs)

# 对每个统计维度，计算 |test_stat - matched_train_stat| 的均值
stat_names = list(test_stats.keys())

print(f"\n{'统计维度':<20} {'SeqMatch 距离':>14} {'PredMatch 距离':>14} {'PredMatch更近?':>14}")
print("=" * 66)

wins = {}
for name in stat_names:
    test_s = test_stats[name]  # [N_test, D]
    train_s = extract_stats(train_seqs)[name]  # [N_train, D]

    seq_dist_all = []
    pred_dist_all = []

    for d in range(D):
        seq_idx = seq_tk_idx[d, :, 0]    # [N_test]
        pred_idx = pred_tk_idx[d, :, 0]  # [N_test]

        seq_matched_stat = train_s[seq_idx, d]    # [N_test]
        pred_matched_stat = train_s[pred_idx, d]  # [N_test]
        test_stat = test_s[:, d]                  # [N_test]

        seq_dist_all.append(np.abs(test_stat - seq_matched_stat))
        pred_dist_all.append(np.abs(test_stat - pred_matched_stat))

    seq_mean = np.mean(seq_dist_all)
    pred_mean = np.mean(pred_dist_all)
    better = "<<< YES" if pred_mean < seq_mean else ""
    wins[name] = pred_mean < seq_mean

    print(f"{name:<20} {seq_mean:>14.6f} {pred_mean:>14.6f} {better:>14}")

print(f"\nPredMatch 更近的统计维度: {sum(wins.values())}/{len(wins)}")
print("胜出维度:", [k for k, v in wins.items() if v])
