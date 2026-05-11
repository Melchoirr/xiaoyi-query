"""绘制测试集上多模型预测对比图。

每个通道一行，展示 seq_len 的输入上下文 + pred_len 的预测曲线。
选取 3 个代表性样本（前期/中期/后期）。
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

CHANNELS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


def main():
    parser = argparse.ArgumentParser(description='多模型预测对比图')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--models', type=str, default='DLinear,PatchTST',
                        help='逗号分隔的基础模型名')
    parser.add_argument('--fusion_model', type=str, default='XGBFusion',
                        help='融合模型名')
    parser.add_argument('--result_path', type=str, default='./outputs/results')
    parser.add_argument('--raw_path', type=str, default='./dataset/ETTh1.csv')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径，默认 result_path/fusion_comparison.png')
    args = parser.parse_args()

    setting_tpl = f'{{model}}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'

    if args.output is None:
        args.output = os.path.join(args.result_path, 'fusion_comparison.png')

    model_names = [m.strip() for m in args.models.split(',')]
    all_models = model_names + [args.fusion_model]

    # ---- 加载原始数据 (标准化后，与模型输入一致) ----
    df_raw = pd.read_csv(args.raw_path)
    df_data = df_raw[CHANNELS].values

    train_end = 12 * 30 * 24       # 8640
    val_end = train_end + 4 * 30 * 24  # 11520

    scaler = StandardScaler()
    scaler.fit(df_data[:train_end])
    data_scaled = scaler.transform(df_data)

    # test set 在 data_scaled 中的起始位置
    test_border1 = val_end - args.seq_len

    # ---- 加载预测结果 ----
    preds = {}
    for m in all_models:
        s = setting_tpl.replace('{model}', m)
        pred_path = os.path.join(args.result_path, s, 'pred.npy')
        preds[m] = np.load(pred_path)

    s_fusion = setting_tpl.replace('{model}', args.fusion_model)
    true = np.load(os.path.join(args.result_path, s_fusion, 'true.npy'))

    N_test = true.shape[0]

    # ---- 选 3 个样本 ----
    sample_indices = [0, N_test // 2, N_test - 1]

    fig, axes = plt.subplots(len(CHANNELS), len(sample_indices),
                             figsize=(6 * len(sample_indices), 2.5 * len(CHANNELS)),
                             sharex='col')

    # 动态生成颜色
    base_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b',
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = {'Ground Truth': '#333333'}
    for i, m in enumerate(model_names):
        colors[m] = base_colors[i % len(base_colors)]
    colors[args.fusion_model] = '#2ca02c'

    for col_idx, sample_i in enumerate(sample_indices):
        abs_start = test_border1 + sample_i
        seq_data = data_scaled[abs_start: abs_start + args.seq_len]
        true_data = data_scaled[abs_start + args.seq_len: abs_start + args.seq_len + args.pred_len]

        x_seq = np.arange(args.seq_len)
        x_pred = np.arange(args.seq_len, args.seq_len + args.pred_len)

        for ch_idx, ch_name in enumerate(CHANNELS):
            ax = axes[ch_idx, col_idx]

            # 输入上下文 (灰色)
            ax.plot(x_seq, seq_data[:, ch_idx], color='#aaaaaa', linewidth=0.8, alpha=0.7)

            # 真实值
            ax.plot(x_pred, true_data[:, ch_idx],
                    color=colors['Ground Truth'], linewidth=1.5, label='Ground Truth')

            # 各模型预测
            for m_name in all_models:
                ax.plot(x_pred, preds[m_name][sample_i, :, ch_idx],
                        color=colors[m_name], linewidth=1.2, alpha=0.85, label=m_name)

            # 分界线
            ax.axvline(x=args.seq_len, color='#cccccc', linestyle='--', linewidth=0.8)

            if col_idx == 0:
                ax.set_ylabel(ch_name, fontsize=10)
            if ch_idx == 0:
                ax.set_title(f'Sample {sample_i}', fontsize=11)
            if ch_idx == len(CHANNELS) - 1:
                ax.set_xlabel('Time step', fontsize=9)

            ax.tick_params(labelsize=8)

    # 图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(all_models) + 1, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'图片已保存到: {args.output}')
    plt.close()


if __name__ == '__main__':
    main()
