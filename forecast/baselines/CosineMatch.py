"""
逐维度 Top-K MSE 加权匹配预测：
对每个维度独立在训练集中找 MSE 最小的 K 个匹配，
用距离倒数加权平均生成预测。
支持 --flags train,val,test 生成多集合预测，兼容 fusion stacking 目录结构。

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
    """加载ETTh1数据，使用原始固定划分，返回归一化和原始两套数据"""
    df_raw = pd.read_csv(os.path.join(root_path, 'ETTh1.csv'))

    border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
    border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]

    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    col_names = list(cols_data)
    raw_data = df_data.values

    scaler = StandardScaler()
    train_data = raw_data[border1s[0]:border2s[0]]
    scaler.fit(train_data)
    norm_data = scaler.transform(raw_data)

    splits_norm, splits_raw = {}, {}
    for i, flag in enumerate(['train', 'val', 'test']):
        splits_norm[flag] = norm_data[border1s[i]:border2s[i]]
        splits_raw[flag] = raw_data[border1s[i]:border2s[i]]

    return splits_norm, splits_raw, col_names, scaler


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


def match_sequences(target_seqs, train_seqs, train_preds, exclude_self=False,
                    top_k=5, batch_size=512,
                    target_seqs_match=None, train_seqs_match=None,
                    return_details=False):
    """逐维度独立 Top-K MSE 加权匹配预测（torch 并行版）。

    对每个维度，用展开公式一次算出 MSE 距离矩阵，torch.topk 取最近邻，
    距离倒数加权平均生成预测。

    Args:
        target_seqs: [N_target, seq_len, D] 归一化序列，用于生成预测值
        train_seqs:  [N_train, seq_len, D]  归一化序列，用于生成预测值
        train_preds: [N_train, pred_len, D] 归一化预测值
        exclude_self: leave-one-out，避免信息泄露
        top_k: 取前 K 个最近邻加权平均
        batch_size: 每批处理的 target 数量，控制显存
        target_seqs_match: [N_target, seq_len, D] 用于距离计算的序列（默认同 target_seqs）
        train_seqs_match:  [N_train, seq_len, D]  用于距离计算的序列（默认同 train_seqs）
        return_details: 是否返回 top-k 索引和距离

    Returns:
        matched_preds: [N_target, pred_len, D]
        若 return_details=True，额外返回:
            all_tk_idx:   [D, N_target, top_k] 每维度每样本的 top-k 训练集索引
            all_tk_dists: [D, N_target, top_k] 对应距离
            all_weights:  [D, N_target, top_k] 对应权重
    """
    device = _get_device()
    if target_seqs_match is None:
        target_seqs_match = target_seqs
    if train_seqs_match is None:
        train_seqs_match = train_seqs

    N_target, seq_len, D = target_seqs.shape
    pred_len = train_preds.shape[1]
    N_train = train_seqs.shape[0]

    matched_preds = np.zeros((N_target, pred_len, D))
    if return_details:
        all_tk_idx = np.zeros((D, N_target, top_k), dtype=int)
        all_tk_dists = np.zeros((D, N_target, top_k))
        all_weights = np.zeros((D, N_target, top_k))

    for d in range(D):
        # 距离计算用 match 序列，预测值用归一化序列
        T_m = torch.from_numpy(train_seqs_match[:, :, d]).float().to(device)  # [N_train, seq_len]
        T_pred = torch.from_numpy(train_preds[:, :, d]).float().to(device)    # [N_train, pred_len]
        T_m_sq = (T_m ** 2).sum(dim=1)  # [N_train]

        for start in range(0, N_target, batch_size):
            end = min(start + batch_size, N_target)
            if d == 0 and start % (batch_size * 4) == 0:
                print(f"  {start}/{N_target}")

            Q_m = torch.from_numpy(target_seqs_match[start:end, :, d]).float().to(device)  # [B, seq_len]
            B = Q_m.shape[0]

            # 展开公式: ||q-t||^2 = ||q||^2 + ||t||^2 - 2*q·t
            Q_m_sq = (Q_m ** 2).sum(dim=1, keepdim=True)                # [B, 1]
            dists = Q_m_sq + T_m_sq.unsqueeze(0) - 2.0 * (Q_m @ T_m.T)  # [B, N_train]

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
            gathered = T_pred[tk_idx.reshape(-1)].reshape(B, top_k, pred_len)
            weighted = (weights.unsqueeze(-1) * gathered).sum(dim=1) # [B, pred_len]

            matched_preds[start:end, :, d] = weighted.cpu().numpy()

            if return_details:
                all_tk_idx[d, start:end] = tk_idx.cpu().numpy()
                all_tk_dists[d, start:end] = tk_dists.cpu().numpy()
                all_weights[d, start:end] = weights.cpu().numpy()

    if return_details:
        return matched_preds, all_tk_idx, all_tk_dists, all_weights
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
        args.output_dir = f'forecast/results/CosineMatch_ETTh1_M_sl{args.seq_len}_pl{args.pred_len}'
    os.makedirs(args.output_dir, exist_ok=True)

    flags = args.flags.split(',')
    print(f"加载数据... flags={flags}")
    splits_norm, splits_raw, col_names, scaler = load_etth1_data(
        args.root_path, args.seq_len, args.pred_len)
    n_dims = len(col_names)

    # 构建训练集序列（归一化用于预测值，原始值用于距离匹配）
    print("构建训练集序列...")
    train_seqs, train_preds = build_sequences(splits_norm['train'], args.seq_len, args.pred_len)
    train_seqs_raw, _ = build_sequences(splits_raw['train'], args.seq_len, args.pred_len)
    print(f"训练集序列数: {len(train_seqs)}")

    for flag in flags:
        print(f"\n{'='*40}")
        print(f"处理 {flag} 集...")

        target_seqs, target_preds = build_sequences(splits_norm[flag], args.seq_len, args.pred_len)
        target_seqs_raw, _ = build_sequences(splits_raw[flag], args.seq_len, args.pred_len)
        print(f"{flag} 集序列数: {len(target_seqs)}")

        # train 集匹配 train 集自身时排除自身，避免信息泄露
        exclude_self = (flag == 'train')
        matched_preds = match_sequences(
            target_seqs, train_seqs, train_preds,
            exclude_self=exclude_self, top_k=args.match_top_k,
            target_seqs_match=target_seqs_raw, train_seqs_match=train_seqs_raw)

        mae, mse, rmse, mape, mspe = metric(matched_preds, target_preds)
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

        # 保存：test直接存根目录，train/val存子目录
        if flag == 'test':
            save_dir = args.output_dir
        else:
            save_dir = os.path.join(args.output_dir, flag)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'pred.npy'), matched_preds)
        np.save(os.path.join(save_dir, 'true.npy'), target_preds)
        np.save(os.path.join(save_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        print(f"  已保存到: {save_dir}")

        setting = f"CosineMatch_ETTh1_M_sl{args.seq_len}_pl{args.pred_len}"
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"RESULT|{ts}|{setting}|{flag}|mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}")

    print("完成！")




if __name__ == '__main__':
    main()
