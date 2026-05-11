"""
逐维度 Top-K MSE 加权匹配预测（pred_len 匹配版）：
直接用目标的 pred_len 窗口在训练集 pred_len 窗口中找 MSE 最小的 K 个匹配，
用距离倒数加权平均生成预测。

与 CosineMatch 的区别：CosineMatch 用 seq_len（历史）匹配，本方法用 pred_len（未来）匹配。
这是一个 oracle/上界实验——测试集已知真实值才能匹配，用于衡量"训练集中存在多相似的未来模式"。

train 集使用 leave-one-out：排除自身匹配，避免信息泄露。
"""

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import os

from forecast.utils.metrics import metric


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


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def match_by_pred(target_preds, train_preds, exclude_self=False,
                  top_k=5, batch_size=512):
    """逐维度独立 Top-K MSE 加权匹配（用 pred_len 窗口匹配）。

    用目标的真实 pred_len 在训练集 pred_len 中找最近邻，加权平均生成预测。

    Args:
        target_preds: [N_target, pred_len, D] — 目标的真实未来值
        train_preds:  [N_train, pred_len, D]  — 训练集的真实未来值
        exclude_self: leave-one-out
        top_k: 取前 K 个最近邻加权平均
        batch_size: 每批处理的 target 数量
    """
    device = _get_device()
    N_target, pred_len, D = target_preds.shape
    N_train = train_preds.shape[0]

    matched_preds = np.zeros((N_target, pred_len, D))

    for d in range(D):
        T = torch.from_numpy(train_preds[:, :, d]).float().to(device)  # [N_train, pred_len]
        T_sq = (T ** 2).sum(dim=1)  # [N_train]

        for start in range(0, N_target, batch_size):
            end = min(start + batch_size, N_target)
            if d == 0 and start % (batch_size * 4) == 0:
                print(f"  {start}/{N_target}")

            Q = torch.from_numpy(target_preds[start:end, :, d]).float().to(device)  # [B, pred_len]
            B = Q.shape[0]

            # MSE 距离: ||q-t||^2 = ||q||^2 + ||t||^2 - 2*q·t
            Q_sq = (Q ** 2).sum(dim=1, keepdim=True)                # [B, 1]
            dists = Q_sq + T_sq.unsqueeze(0) - 2.0 * (Q @ T.T)     # [B, N_train]

            if exclude_self:
                idx = torch.arange(start, end, device=device)
                mask = idx < N_train
                if mask.any():
                    dists[mask, idx[mask]] = float('inf')

            # Top-K 最小距离
            tk_dists, tk_idx = torch.topk(dists, top_k, dim=1, largest=False)  # [B, K]

            # 距离倒数加权
            weights = 1.0 / (tk_dists + 1e-8)                       # [B, K]
            weights = weights / weights.sum(dim=1, keepdim=True)     # [B, K]
            gathered = T[tk_idx.reshape(-1)].reshape(B, top_k, pred_len)
            weighted = (weights.unsqueeze(-1) * gathered).sum(dim=1) # [B, pred_len]

            matched_preds[start:end, :, d] = weighted.cpu().numpy()

    return matched_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='dataset')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--flags', type=str, default='test,train,val',
                        help='逗号分隔: test,train,val')
    parser.add_argument('--match_top_k', type=int, default=5,
                        help='匹配时取前K个最近邻加权平均')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认自动生成')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'outputs/results/PredMatch_ETTh1_M_sl{args.seq_len}_pl{args.pred_len}'
    os.makedirs(args.output_dir, exist_ok=True)

    flags = args.flags.split(',')
    print(f"加载数据... flags={flags}")
    splits, col_names, scaler = load_etth1_data(args.root_path, args.seq_len, args.pred_len)

    # 构建训练集序列（作为检索库）
    print("构建训练集序列...")
    train_seqs, train_preds = build_sequences(splits['train'], args.seq_len, args.pred_len)
    print(f"训练集序列数: {len(train_seqs)}")

    for flag in flags:
        print(f"\n{'='*40}")
        print(f"处理 {flag} 集...")

        target_seqs, target_preds = build_sequences(splits[flag], args.seq_len, args.pred_len)
        print(f"{flag} 集序列数: {len(target_seqs)}")

        # train 集匹配自身时排除自身
        exclude_self = (flag == 'train')

        # 核心区别：用 target_preds（真实未来）去匹配 train_preds
        matched_preds = match_by_pred(
            target_preds, train_preds,
            exclude_self=exclude_self, top_k=args.match_top_k)

        mae, mse, rmse, mape, mspe = metric(matched_preds, target_preds)
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

        # 保存
        if flag == 'test':
            save_dir = args.output_dir
        else:
            save_dir = os.path.join(args.output_dir, flag)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'pred.npy'), matched_preds)
        np.save(os.path.join(save_dir, 'true.npy'), target_preds)
        np.save(os.path.join(save_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        print(f"  已保存到: {save_dir}")

        setting = f"PredMatch_ETTh1_M_sl{args.seq_len}_pl{args.pred_len}"
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"RESULT|{ts}|{setting}|{flag}|mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}")

    print("完成！")


if __name__ == '__main__':
    main()
