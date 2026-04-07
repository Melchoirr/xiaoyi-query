"""ETTh1 周期性与相位差分析：PSD + 互相关 + 交叉谱三件套。"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

df = pd.read_csv("dataset/ETTh1.csv", parse_dates=["date"])
cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# 仅用 train 集分析，避免数据泄露
train_end = 12 * 30 * 24  # 8640
train_mean = df[cols].iloc[:train_end].mean()
train_std = df[cols].iloc[:train_end].std()
data = ((df[cols].iloc[:train_end] - train_mean) / train_std).values  # (8640, 7)

fs = 1.0  # 采样频率: 1 sample/hour

# ============================================================
# 图1: 功率谱密度 (PSD) — 找主周期
# ============================================================
fig1, axes1 = plt.subplots(4, 2, figsize=(16, 14))
axes1 = axes1.flatten()

for i, col in enumerate(cols):
    ax = axes1[i]
    freqs, psd = signal.welch(data[:, i], fs=fs, nperseg=1024)
    periods = 1.0 / (freqs[1:] + 1e-12)  # 转为周期(小时)

    ax.semilogy(periods, psd[1:], linewidth=0.8)
    ax.set_xlim(0, 200)
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("PSD")
    ax.set_title(col, fontweight="bold")
    # 标注关键周期
    for p, label in [(24, "24h"), (12, "12h"), (168, "168h(7d)")]:
        ax.axvline(p, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(p + 1, ax.get_ylim()[1] * 0.3, label, fontsize=7, color="red")

axes1[-1].set_visible(False)
fig1.suptitle("ETTh1 Train — Power Spectral Density (主周期识别)", fontsize=14)
plt.tight_layout()
fig1.savefig("forecast/scripts/etth1_psd.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved etth1_psd.png")

# ============================================================
# 图2: 互相关 — 全局 lag (以 OT 为参考)
# ============================================================
ref_idx = cols.index("OT")
ref = data[:, ref_idx]
max_lag = 200

fig2, axes2 = plt.subplots(3, 2, figsize=(16, 10))
axes2 = axes2.flatten()

other_cols = [c for c in cols if c != "OT"]
for i, col in enumerate(other_cols):
    ax = axes2[i]
    x = data[:, cols.index(col)]
    # 归一化互相关
    cc = np.correlate(x - x.mean(), ref - ref.mean(), mode="full")
    cc /= (np.std(x) * np.std(ref) * len(x))
    lags = np.arange(-len(x) + 1, len(x))
    mask = np.abs(lags) <= max_lag
    ax.plot(lags[mask], cc[mask], linewidth=0.8)
    peak_lag = lags[mask][np.argmax(cc[mask])]
    ax.axvline(peak_lag, color="red", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_title(f"{col} vs OT  (peak lag = {peak_lag}h)", fontweight="bold")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Cross-correlation")

fig2.suptitle("ETTh1 Train — Cross-correlation with OT (相位差/滞后)", fontsize=14)
plt.tight_layout()
fig2.savefig("forecast/scripts/etth1_xcorr.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved etth1_xcorr.png")

# ============================================================
# 图3: 交叉谱 — Coherence + Phase (以 OT 为参考)
# ============================================================
fig3, axes3 = plt.subplots(6, 2, figsize=(16, 20))

for i, col in enumerate(other_cols):
    x = data[:, cols.index(col)]
    freqs, coh = signal.coherence(x, ref, fs=fs, nperseg=1024)
    freqs_csd, csd = signal.csd(x, ref, fs=fs, nperseg=1024)
    phase = np.angle(csd, deg=True)  # 相位差(度)
    periods = 1.0 / (freqs[1:] + 1e-12)

    # Coherence
    ax_c = axes3[i, 0]
    ax_c.plot(periods, coh[1:], linewidth=0.8)
    ax_c.set_xlim(0, 200)
    ax_c.set_ylabel("Coherence")
    ax_c.set_title(f"{col} vs OT — Coherence", fontweight="bold", fontsize=10)
    for p in [12, 24, 168]:
        ax_c.axvline(p, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    # Phase
    ax_p = axes3[i, 1]
    ax_p.scatter(periods, phase[1:], s=2, alpha=0.5)
    ax_p.set_xlim(0, 200)
    ax_p.set_ylim(-180, 180)
    ax_p.set_ylabel("Phase diff (°)")
    ax_p.set_title(f"{col} vs OT — Phase", fontweight="bold", fontsize=10)
    for p in [12, 24, 168]:
        ax_p.axvline(p, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        # 标注关键周期处的相位差
        idx_p = np.argmin(np.abs(1.0 / (freqs[1:] + 1e-12) - p))
        ph_val = phase[1:][idx_p]
        lag_h = ph_val / 360 * p  # 转换为小时
        ax_p.annotate(f"{p}h: {ph_val:.0f}°({lag_h:.1f}h)",
                      xy=(p, ph_val), fontsize=7, color="red",
                      xytext=(p + 8, ph_val + 20),
                      arrowprops=dict(arrowstyle="->", color="red", lw=0.5))

axes3[-1, 0].set_xlabel("Period (hours)")
axes3[-1, 1].set_xlabel("Period (hours)")

fig3.suptitle("ETTh1 Train — Cross-spectral Analysis with OT (频域相位差)", fontsize=14)
plt.tight_layout()
fig3.savefig("forecast/scripts/etth1_cross_spectral.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved etth1_cross_spectral.png")

# ============================================================
# 图4: 汇总表 — 关键周期处的 coherence 和 phase lag
# ============================================================
print("\n=== 关键周期处各变量与 OT 的 Coherence & Phase Lag ===")
print(f"{'Variable':<8} | {'12h Coh':>8} {'12h Lag':>8} | {'24h Coh':>8} {'24h Lag':>8} | {'168h Coh':>8} {'168h Lag':>8}")
print("-" * 80)
for col in other_cols:
    x = data[:, cols.index(col)]
    freqs, coh = signal.coherence(x, ref, fs=fs, nperseg=1024)
    _, csd = signal.csd(x, ref, fs=fs, nperseg=1024)
    phase = np.angle(csd, deg=True)
    periods_arr = 1.0 / (freqs[1:] + 1e-12)

    parts = []
    for p in [12, 24, 168]:
        idx_p = np.argmin(np.abs(periods_arr - p))
        c = coh[1:][idx_p]
        ph = phase[1:][idx_p]
        lag_h = ph / 360 * p
        parts.append(f"{c:>8.3f} {lag_h:>7.1f}h")
    print(f"{col:<8} | {' | '.join(parts)}")
