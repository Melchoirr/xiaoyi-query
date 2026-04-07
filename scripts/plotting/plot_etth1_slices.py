"""ETTh1 归一化后每 480 点切片，每张图放 6 个切片，输出多张图。"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("dataset/ETTh1.csv", parse_dates=["date"])
cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

train_end = 12 * 30 * 24
val_end   = train_end + 4 * 30 * 24

train_mean = df[cols].iloc[:train_end].mean()
train_std  = df[cols].iloc[:train_end].std()
df[cols] = (df[cols] - train_mean) / train_std

step = 480
n_slices = len(df) // step
per_page = 6  # 每张图放 6 个切片
n_pages = int(np.ceil(n_slices / per_page))
colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))

outdir = "outputs/figures/data_analysis/etth1_slices"
os.makedirs(outdir, exist_ok=True)

for page in range(n_pages):
    s = page * per_page
    e = min(s + per_page, n_slices)
    n_sub = e - s

    fig, axes = plt.subplots(n_sub, 1, figsize=(14, n_sub * 2.8), sharex=False, sharey=True)
    if n_sub == 1:
        axes = [axes]

    for idx, i in enumerate(range(s, e)):
        ax = axes[idx]
        start = i * step
        end = start + step
        chunk = df.iloc[start:end]

        for j, col in enumerate(cols):
            ax.plot(chunk["date"], chunk[col], linewidth=0.6, color=colors[j], label=col if idx == 0 else None)

        if end <= train_end:
            tag, tc = "Train", "green"
        elif start >= val_end:
            tag, tc = "Test", "orange"
        elif start >= train_end:
            tag, tc = "Val", "purple"
        else:
            tag, tc = "boundary", "red"

        ax.set_title(f"Slice #{i}   {chunk['date'].iloc[0].strftime('%Y-%m-%d %H:%M')} ~ "
                     f"{chunk['date'].iloc[-1].strftime('%Y-%m-%d %H:%M')}   [{tag}]",
                     fontsize=9, color=tc, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:00"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))

    handles = [plt.Line2D([0], [0], color=colors[j], lw=1.5) for j in range(len(cols))]
    fig.legend(handles, cols, loc="upper right", fontsize=8, ncol=len(cols))
    fig.suptitle(f"ETTh1 Normalized — 480-pt Slices  (Page {page+1}/{n_pages})", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"{outdir}/page_{page+1:02d}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

print(f"\nDone — {n_pages} images in {outdir}/")
