"""
MSE最小匹配预测：
对目标集每个序列，在训练集中找MSE最小的历史序列，用其后续pred_len作为预测。
支持 --flags train,val,test 生成多集合预测，兼容 fusion stacking 目录结构。

train 集使用 leave-one-out：排除自身匹配，避免信息泄露。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

from forecast.utils.metrics import metric

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_etth1_data(root_path, seq_len=512, pred_len=96):
    """加载ETTh1数据，使用原始固定划分，返回全量归一化数据和边界"""
    df_raw = pd.read_csv(os.path.join(root_path, 'ETTh1.csv'))

    border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
    border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]

    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    col_names = list(cols_data)

    scaler = StandardScaler()
    train_data = df_data.values[border1s[0]:border2s[0]]
    scaler.fit(train_data)
    data = scaler.transform(df_data.values)

    splits = {}
    for i, flag in enumerate(['train', 'val', 'test']):
        splits[flag] = data[border1s[i]:border2s[i]]

    return splits, col_names, scaler


def build_sequences(data, seq_len, pred_len):
    """从连续数据构建 (seq, pred) 对"""
    n = len(data) - seq_len - pred_len + 1
    seqs = np.array([data[i:i+seq_len] for i in range(n)])
    preds = np.array([data[i+seq_len:i+seq_len+pred_len] for i in range(n)])
    return seqs, preds


def match_sequences(target_seqs, train_seqs, train_preds, exclude_self=False):
    """对target_seqs中每个序列，在train_seqs中找MSE最小的，返回对应的train_preds。

    Args:
        exclude_self: 当 target 和 train 来自同一集合时设为 True，
                      排除自身匹配（leave-one-out），避免信息泄露。
    """
    N_target = len(target_seqs)
    N_train = len(train_seqs)
    T_flat = train_seqs.reshape(N_train, -1)  # [N_train, seq_len*D]

    best_indices = np.zeros(N_target, dtype=int)
    all_dists = np.zeros((N_target, N_train))

    for i in range(N_target):
        if i % 500 == 0:
            print(f"  {i}/{N_target}")
        t = target_seqs[i].flatten()
        dists = np.mean((T_flat - t) ** 2, axis=1)

        if exclude_self and i < N_train:
            dists[i] = np.inf

        all_dists[i] = dists
        best_indices[i] = np.argmin(dists)

    matched_preds = train_preds[best_indices]
    return matched_preds, best_indices, all_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='dataset')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--flags', type=str, default='test,train,val',
                        help='逗号分隔: test,train,val')
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认自动生成')
    parser.add_argument('--plot', action='store_true', help='是否绘图')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'forecast/results/CosineMatch_ETTh1_M_sl{args.seq_len}_pl{args.pred_len}'
    os.makedirs(args.output_dir, exist_ok=True)

    flags = args.flags.split(',')
    print(f"加载数据... flags={flags}")
    splits, col_names, scaler = load_etth1_data(args.root_path, args.seq_len, args.pred_len)
    n_dims = len(col_names)

    # 构建训练集序列（始终需要，作为检索库）
    print("构建训练集序列...")
    train_seqs, train_preds = build_sequences(splits['train'], args.seq_len, args.pred_len)
    print(f"训练集序列数: {len(train_seqs)}")

    test_mse, test_mae = None, None

    for flag in flags:
        print(f"\n{'='*40}")
        print(f"处理 {flag} 集...")

        target_seqs, target_preds = build_sequences(splits[flag], args.seq_len, args.pred_len)
        print(f"{flag} 集序列数: {len(target_seqs)}")

        # train 集匹配 train 集自身时排除自身，避免信息泄露
        exclude_self = (flag == 'train')
        matched_preds, best_indices, all_dists = match_sequences(
            target_seqs, train_seqs, train_preds, exclude_self=exclude_self)

        mae, mse, rmse, mape, mspe = metric(matched_preds, target_preds)
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

        if flag == 'test':
            test_mse, test_mae = mse, mae

        # 保存：test直接存根目录，train/val存子目录
        if flag == 'test':
            save_dir = args.output_dir
        else:
            save_dir = os.path.join(args.output_dir, flag)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'pred.npy'), matched_preds)
        np.save(os.path.join(save_dir, 'true.npy'), target_preds)
        np.save(os.path.join(save_dir, 'metrics.npy'), np.array([mae, mse]))
        print(f"  已保存到: {save_dir}")

        # 绘图仅对test集
        if flag == 'test' and args.plot:
            _plot_comparison(args, target_seqs, target_preds, train_seqs, train_preds,
                             best_indices, all_dists, col_names, n_dims, mse, mae)

    # 始终打印 test 集结果
    if test_mse is not None:
        print(f"\nRESULT|CosineMatch_ETTh1_M_sl{args.seq_len}_pl{args.pred_len}|test|"
              f"mse={test_mse:.6f}|mae={test_mae:.6f}")
    print("完成！")


def _plot_comparison(args, test_seqs, test_preds, train_seqs, train_preds,
                     best_indices, all_dists, col_names, n_dims, mse, mae):
    """绘制对比图和top-k匹配图"""
    N_test = len(test_seqs)
    sample_indices = np.linspace(0, N_test - 1, args.n_samples, dtype=int)

    # 图1: 对比图
    fig, axes = plt.subplots(n_dims, args.n_samples, figsize=(7 * args.n_samples, 3.5 * n_dims))
    fig.suptitle(f'MSE最小匹配预测 (seq_len={args.seq_len}, pred_len={args.pred_len})\n'
                 f'MSE={mse:.4f}, MAE={mae:.4f}', fontsize=14, y=1.01)

    for col_idx, si in enumerate(sample_indices):
        bi = best_indices[si]
        for row_idx in range(n_dims):
            ax = axes[row_idx, col_idx] if args.n_samples > 1 else axes[row_idx]
            x_seq = np.arange(args.seq_len)
            x_pred = np.arange(args.seq_len, args.seq_len + args.pred_len)

            ax.plot(x_seq, test_seqs[si, :, row_idx], color='blue', alpha=0.8, label='Test Seq', linewidth=1)
            ax.plot(x_pred, test_preds[si, :, row_idx], color='blue', linestyle='--', alpha=0.8, label='Test GT', linewidth=1)
            ax.plot(x_seq, train_seqs[bi, :, row_idx], color='red', alpha=0.6, label='Hist Seq', linewidth=1)
            ax.plot(x_pred, train_preds[bi, :, row_idx], color='red', linestyle='--', alpha=0.6, label='Hist Pred', linewidth=1)
            ax.axvline(x=args.seq_len, color='gray', linestyle=':', alpha=0.5)

            if row_idx == 0:
                ax.set_title(f'Sample {col_idx} (test#{si}, mse={all_dists[si, bi]:.4f})', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(col_names[row_idx], fontsize=9)
            if row_idx == 0 and col_idx == args.n_samples - 1:
                ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 图2: Top-K
    for col_idx, si in enumerate(sample_indices):
        top_k_indices = np.argsort(all_dists[si])[:args.top_k]
        top_k_dists = all_dists[si, top_k_indices]

        fig2, axes2 = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims))
        fig2.suptitle(f'Sample {col_idx} (test#{si}) — Top {args.top_k} MSE最小匹配', fontsize=13, y=1.01)

        for row_idx in range(n_dims):
            ax = axes2[row_idx]
            x_seq = np.arange(args.seq_len)
            x_pred = np.arange(args.seq_len, args.seq_len + args.pred_len)

            ax.plot(x_seq, test_seqs[si, :, row_idx], color='blue', linewidth=1.5, label='Test Seq', zorder=10)
            ax.plot(x_pred, test_preds[si, :, row_idx], color='blue', linestyle='--', linewidth=1.5, label='Test GT', zorder=10)

            cmap = plt.cm.Reds
            for rank, ki in enumerate(top_k_indices):
                color = cmap(0.9 - 0.6 * rank / args.top_k)
                alpha = 0.8 - 0.05 * rank
                lbl = f'#{rank+1} (mse={top_k_dists[rank]:.3f})' if rank < 3 else None
                ax.plot(x_seq, train_seqs[ki, :, row_idx], color=color, alpha=alpha, linewidth=0.8)
                ax.plot(x_pred, train_preds[ki, :, row_idx], color=color, alpha=alpha, linewidth=0.8, linestyle='--', label=lbl)

            ax.axvline(x=args.seq_len, color='gray', linestyle=':', alpha=0.5)
            ax.set_ylabel(col_names[row_idx], fontsize=9)
            if row_idx == 0:
                ax.legend(fontsize=7, loc='upper right', ncol=2)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'top{args.top_k}_sample{col_idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
