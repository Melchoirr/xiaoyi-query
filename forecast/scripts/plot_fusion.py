"""绘制测试集上 DLinear / PatchTST / XGBFusion 预测对比图。

每个通道一行，展示 seq_len(336) 的输入上下文 + pred_len(96) 的预测曲线。
选取 3 个代表性样本（前期/中期/后期）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---- 配置 ----
SEQ_LEN = 336
PRED_LEN = 96
RESULT_BASE = './forecast/results'
SETTING_TPL = '{model}_ETTh1_M_sl336_pl96'
MODELS = ['DLinear', 'PatchTST']
RAW_PATH = './dataset/ETTh1.csv'
CHANNELS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# ---- 加载原始数据 (标准化后，与模型输入一致) ----
df_raw = pd.read_csv(RAW_PATH)
df_data = df_raw[CHANNELS].values

train_end = 12 * 30 * 24       # 8640
val_end = train_end + 4 * 30 * 24  # 11520

scaler = StandardScaler()
scaler.fit(df_data[:train_end])
data_scaled = scaler.transform(df_data)

# test set 在 data_scaled 中的起始位置
test_border1 = val_end - SEQ_LEN  # 11520 - 336 = 11184
# data_x = data_scaled[test_border1 : test_border2] 其中 test_border2 = val_end + 4*30*24

# ---- 加载预测结果 ----
preds = {}
for m in MODELS:
    s = SETTING_TPL.replace('{model}', m)
    preds[m] = np.load(f'{RESULT_BASE}/{s}/pred.npy')  # [N, 96, 7]

s_fusion = SETTING_TPL.replace('{model}', 'XGBFusion')
preds['XGBFusion'] = np.load(f'{RESULT_BASE}/{s_fusion}/pred.npy')
true = np.load(f'{RESULT_BASE}/{s_fusion}/true.npy')

N_test = true.shape[0]  # 2785

# ---- 选 3 个样本 ----
sample_indices = [0, N_test // 2, N_test - 1]

fig, axes = plt.subplots(len(CHANNELS), len(sample_indices),
                         figsize=(6 * len(sample_indices), 2.5 * len(CHANNELS)),
                         sharex='col')

colors = {
    'Ground Truth': '#333333',
    'DLinear': '#1f77b4',
    'PatchTST': '#ff7f0e',
    'XGBFusion': '#2ca02c',
}

for col_idx, sample_i in enumerate(sample_indices):
    # 原始数据中的绝对位置
    abs_start = test_border1 + sample_i  # seq 起始
    seq_data = data_scaled[abs_start: abs_start + SEQ_LEN]        # [336, 7]
    true_data = data_scaled[abs_start + SEQ_LEN: abs_start + SEQ_LEN + PRED_LEN]  # [96, 7]

    x_seq = np.arange(SEQ_LEN)
    x_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

    for ch_idx, ch_name in enumerate(CHANNELS):
        ax = axes[ch_idx, col_idx]

        # 输入上下文 (灰色)
        ax.plot(x_seq, seq_data[:, ch_idx], color='#aaaaaa', linewidth=0.8, alpha=0.7)

        # 真实值
        ax.plot(x_pred, true_data[:, ch_idx],
                color=colors['Ground Truth'], linewidth=1.5, label='Ground Truth')

        # 各模型预测
        for m_name in ['DLinear', 'PatchTST', 'XGBFusion']:
            ax.plot(x_pred, preds[m_name][sample_i, :, ch_idx],
                    color=colors[m_name], linewidth=1.2, alpha=0.85, label=m_name)

        # 分界线
        ax.axvline(x=SEQ_LEN, color='#cccccc', linestyle='--', linewidth=0.8)

        if col_idx == 0:
            ax.set_ylabel(ch_name, fontsize=10)
        if ch_idx == 0:
            ax.set_title(f'Sample {sample_i}', fontsize=11)
        if ch_idx == len(CHANNELS) - 1:
            ax.set_xlabel('Time step', fontsize=9)

        ax.tick_params(labelsize=8)

# 图例
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = './forecast/results/fusion_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'图片已保存到: {out_path}')
plt.close()
