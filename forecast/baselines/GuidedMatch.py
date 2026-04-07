"""
预测引导匹配 (Guided Match)：
用已训练模型（如 DLinear）的预测作为 query，在训练集真实未来中检索最相似的，
返回检索到的训练集未来加权平均作为最终预测。

核心思想：DLinear 的预测（MSE≈0.37）比原始历史序列（MSE≈0.77）更接近真实未来，
用它做 query 检索比用历史序列更准确。
"""

import argparse
import numpy as np
import torch
import os

from forecast.utils.metrics import metric
from forecast.baselines.CosineMatch import load_etth1_data, build_sequences


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def guided_match(guide_preds, train_preds, exclude_self=False,
                 top_k=5, batch_size=512):
    """用引导预测在训练集未来中检索匹配，逐维度独立 Top-K 加权。

    Args:
        guide_preds:  [N_target, pred_len, D] 引导模型的预测（如 DLinear 输出）
        train_preds:  [N_train, pred_len, D]  训练集的真实未来值
        exclude_self: leave-one-out
        top_k: 取前 K 个最近邻加权平均
        batch_size: 每批处理的 target 数量
    """
    device = _get_device()
    N_target, pred_len, D = guide_preds.shape
    N_train = train_preds.shape[0]

    matched_preds = np.zeros((N_target, pred_len, D))

    for d in range(D):
        T = torch.from_numpy(train_preds[:, :, d]).float().to(device)
        T_sq = (T ** 2).sum(dim=1)

        for start in range(0, N_target, batch_size):
            end = min(start + batch_size, N_target)
            if d == 0 and start % (batch_size * 4) == 0:
                print(f"  {start}/{N_target}")

            Q = torch.from_numpy(guide_preds[start:end, :, d]).float().to(device)
            B = Q.shape[0]

            Q_sq = (Q ** 2).sum(dim=1, keepdim=True)
            dists = Q_sq + T_sq.unsqueeze(0) - 2.0 * (Q @ T.T)

            if exclude_self:
                idx = torch.arange(start, end, device=device)
                mask = idx < N_train
                if mask.any():
                    dists[mask, idx[mask]] = float('inf')

            tk_dists, tk_idx = torch.topk(dists, top_k, dim=1, largest=False)

            weights = 1.0 / (tk_dists + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)
            gathered = T[tk_idx.reshape(-1)].reshape(B, top_k, pred_len)
            weighted = (weights.unsqueeze(-1) * gathered).sum(dim=1)

            matched_preds[start:end, :, d] = weighted.cpu().numpy()

    return matched_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='dataset')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--guide_model', type=str, default='DLinear',
                        help='引导模型名称，用于定位其预测结果')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--flags', type=str, default='test,train,val',
                        help='逗号分隔: test,train,val')
    parser.add_argument('--match_top_k', type=int, default=5,
                        help='匹配时取前K个最近邻加权平均')
    parser.add_argument('--result_path', type=str, default='forecast/results')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    guide_setting = f'{args.guide_model}_ETTh1_{args.features}_sl{args.seq_len}_pl{args.pred_len}'
    out_setting = f'GuidedMatch_{args.guide_model}_ETTh1_{args.features}_sl{args.seq_len}_pl{args.pred_len}'

    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_path, out_setting)
    os.makedirs(args.output_dir, exist_ok=True)

    flags = args.flags.split(',')
    print(f"引导模型: {args.guide_model} ({guide_setting})")
    print(f"flags: {flags}")

    # 加载训练集真实未来（检索库）
    print("加载训练集真实未来...")
    splits_norm, _, col_names, _ = load_etth1_data(args.root_path, args.seq_len, args.pred_len)
    _, train_preds = build_sequences(splits_norm['train'], args.seq_len, args.pred_len)
    print(f"训练集序列数: {len(train_preds)}")

    for flag in flags:
        print(f"\n{'='*40}")
        print(f"处理 {flag} 集...")

        # 加载引导模型的预测
        if flag == 'test':
            guide_dir = os.path.join(args.result_path, guide_setting)
        else:
            guide_dir = os.path.join(args.result_path, guide_setting, flag)

        guide_pred_path = os.path.join(guide_dir, 'pred.npy')
        true_path = os.path.join(guide_dir, 'true.npy')

        if not os.path.exists(guide_pred_path):
            print(f"  跳过: {guide_pred_path} 不存在")
            continue

        guide_preds = np.load(guide_pred_path)
        true_preds = np.load(true_path)
        print(f"  引导预测 shape: {guide_preds.shape}")

        # 引导模型自身的指标
        g_mae, g_mse, *_ = metric(guide_preds, true_preds)
        print(f"  引导模型自身: MSE={g_mse:.6f}, MAE={g_mae:.6f}")

        # 用引导预测做匹配
        exclude_self = (flag == 'train')
        matched_preds = guided_match(
            guide_preds, train_preds,
            exclude_self=exclude_self, top_k=args.match_top_k)

        mae, mse, rmse, mape, mspe = metric(matched_preds, true_preds)
        print(f"  引导匹配结果: MSE={mse:.6f}, MAE={mae:.6f}")

        # 保存
        if flag == 'test':
            save_dir = args.output_dir
        else:
            save_dir = os.path.join(args.output_dir, flag)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'pred.npy'), matched_preds)
        np.save(os.path.join(save_dir, 'true.npy'), true_preds)
        np.save(os.path.join(save_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        print(f"  已保存到: {save_dir}")

        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"RESULT|{ts}|{out_setting}|{flag}|mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}")

    print("完成！")


if __name__ == '__main__':
    main()
