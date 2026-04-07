"""绘制 ETTh1 基于 train 集归一化后的 7 个变量序列，竖直虚线标注集合划分。"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/ETTh1.csv", parse_dates=["date"])
cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# ETT 标准划分边界
train_end = 12 * 30 * 24        # 8640
val_end   = train_end + 4 * 30 * 24  # 12960

# 用 train 集的均值和标准差做归一化
train_mean = df[cols].iloc[:train_end].mean()
train_std  = df[cols].iloc[:train_end].std()
df[cols] = (df[cols] - train_mean) / train_std

fig, axes = plt.subplots(7, 1, figsize=(16, 14), sharex=True)

for ax, col in zip(axes, cols):
    ax.plot(df["date"], df[col], linewidth=0.4)
    ax.set_ylabel(col, fontsize=10)
    ax.axvline(df["date"].iloc[train_end], color="red", linestyle="--", linewidth=1, label="Train/Val")
    ax.axvline(df["date"].iloc[val_end],   color="blue", linestyle="--", linewidth=1, label="Val/Test")
    if col == cols[0]:
        ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Date")
fig.suptitle("ETTh1 Normalized by Train Set (Train | Val | Test)", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("outputs/figures/data_analysis/etth1_normalized_series.png", dpi=150)
plt.show()
print("Saved to outputs/figures/data_analysis/etth1_normalized_series.png")
