"""逐 channel 对比 SeqMatch / GuidedMatch / PredMatch 的 MSE/MAE"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from forecast.baselines.CosineMatch import load_etth1_data, build_sequences

splits_norm, _, col_names, _ = load_etth1_data('dataset', 96, 96)
_, true_preds = build_sequences(splits_norm['test'], 96, 96)
D = len(col_names)

methods = {
    'SeqMatch':     'forecast/results/CosineMatch_ETTh1_M_sl96_pl96/pred.npy',
    'GuidedMatch':  'forecast/results/GuidedMatch_DLinear_ETTh1_M_sl96_pl96/pred.npy',
    'DLinear':      'forecast/results/DLinear_ETTh1_M_sl96_pl96/pred.npy',
    'PredMatch':    'forecast/results/PredMatch_ETTh1_M_sl96_pl96/pred.npy',
}

# 加载所有预测
preds = {}
for name, path in methods.items():
    if os.path.exists(path):
        preds[name] = np.load(path)
    else:
        print(f"跳过 {name}: {path} 不存在")

# 打印表头
names = list(preds.keys())
header = f"{'Channel':<10}"
for n in names:
    header += f" {n+' MSE':>14} {n+' MAE':>14}"
print(header)
print("=" * len(header))

# 逐 channel
total_mse = {n: [] for n in names}
total_mae = {n: [] for n in names}

for d in range(D):
    row = f"{col_names[d]:<10}"
    for n in names:
        p = preds[n][:, :, d]
        t = true_preds[:, :, d]
        mse = np.mean((p - t) ** 2)
        mae = np.mean(np.abs(p - t))
        total_mse[n].append(mse)
        total_mae[n].append(mae)
        row += f" {mse:>14.4f} {mae:>14.4f}"
    print(row)

# 全维度平均
print("-" * len(header))
row = f"{'平均':<10}"
for n in names:
    row += f" {np.mean(total_mse[n]):>14.4f} {np.mean(total_mae[n]):>14.4f}"
print(row)
