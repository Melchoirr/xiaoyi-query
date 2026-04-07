"""
分析 SeqMatch vs PredMatch 匹配到的训练序列在时间周期相位上的对齐度。
比较日内相位差 (mod 24) 和周内相位差 (mod 168)。
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from forecast.baselines.CosineMatch import (
    load_etth1_data, build_sequences, match_sequences, _get_device
)

SEQ_LEN = 96
PRED_LEN = 96

# ── 加载数据 ──
print("加载数据...")
splits_norm, splits_raw, col_names, scaler = load_etth1_data('dataset', SEQ_LEN, PRED_LEN)
D = len(col_names)

train_seqs, train_preds = build_sequences(splits_norm['train'], SEQ_LEN, PRED_LEN)
train_seqs_raw, _ = build_sequences(splits_raw['train'], SEQ_LEN, PRED_LEN)
test_seqs, test_preds = build_sequences(splits_norm['test'], SEQ_LEN, PRED_LEN)
test_seqs_raw, _ = build_sequences(splits_raw['test'], SEQ_LEN, PRED_LEN)
N_test = len(test_seqs)
N_train = len(train_seqs)

# ── 计算每个序列的起始时间索引 ──
# train 集起始位置: 0, 1, 2, ... (在原始数据中的偏移)
# test 集起始位置: border1s[2] + 0, 1, 2, ...
# 但我们只需要 mod 周期，所以直接用序列索引即可
# train: 序列 i 的起始位置 = i (在 train split 中)
# test:  序列 i 的起始位置 = border1_test + i
# border1_test = 12*30*24 + 4*30*24 - SEQ_LEN

border1_train = 0
border1_test = 12 * 30 * 24 + 4 * 30 * 24 - SEQ_LEN

train_start_hours = np.arange(N_train) + border1_train  # 在全局时间轴上的位置
test_start_hours = np.arange(N_test) + border1_test

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

# ── 相位差计算 ──
def circular_diff(a, b, period):
    """循环距离: min(|a-b| mod period, period - |a-b| mod period)"""
    diff = np.abs(a - b) % period
    return np.minimum(diff, period - diff)


print(f"\n{'='*70}")
print(f"时间周期相位对齐分析 (SeqMatch vs PredMatch, top-1)")
print(f"{'='*70}")

for period, period_name in [(24, '日周期 (24h)'), (168, '周周期 (168h)')]:
    print(f"\n── {period_name} ──")
    print(f"{'维度':<12} {'SeqMatch 均值':>14} {'PredMatch 均值':>14} {'PredMatch更近':>14}")
    print("-" * 56)

    seq_diffs_all = []
    pred_diffs_all = []

    for d in range(D):
        test_phase = test_start_hours % period           # [N_test]
        seq_train_phase = train_start_hours[seq_tk_idx[d, :, 0]] % period
        pred_train_phase = train_start_hours[pred_tk_idx[d, :, 0]] % period

        seq_phase_diff = circular_diff(test_phase, seq_train_phase, period)
        pred_phase_diff = circular_diff(test_phase, pred_train_phase, period)

        seq_mean = seq_phase_diff.mean()
        pred_mean = pred_phase_diff.mean()
        better = "<<<" if pred_mean < seq_mean else ""

        print(f"{col_names[d]:<12} {seq_mean:>14.2f} {pred_mean:>14.2f} {better:>14}")

        seq_diffs_all.append(seq_phase_diff)
        pred_diffs_all.append(pred_phase_diff)

    seq_total = np.mean(seq_diffs_all)
    pred_total = np.mean(pred_diffs_all)
    better = "<<<" if pred_total < seq_total else ""
    print(f"{'全维度平均':<12} {seq_total:>14.2f} {pred_total:>14.2f} {better:>14}")

    # 分布统计
    seq_flat = np.concatenate(seq_diffs_all)
    pred_flat = np.concatenate(pred_diffs_all)
    print(f"\n  相位差分布 (最大值={period//2}):")
    print(f"  {'':12} {'中位数':>8} {'<={period//8}h 占比':>14} {'==0h 占比':>12}")
    print(f"  {'SeqMatch':12} {np.median(seq_flat):>8.1f} {(seq_flat <= period//8).mean():>14.1%} {(seq_flat == 0).mean():>12.1%}")
    print(f"  {'PredMatch':12} {np.median(pred_flat):>8.1f} {(pred_flat <= period//8).mean():>14.1%} {(pred_flat == 0).mean():>12.1%}")
