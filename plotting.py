"""
静态可视化模块 - 四段线对比波形图

v3.0 核心设计：
四段线布局（从左到右时间顺序）：

  区域 A: Historical Lookback     [t=-seq_len .. -1]
           模型检索到的最相似历史子序列 X_match（来自训练记忆库）

  区域 B: Historical Prediction   [t=-pred_len .. -1]
           X_match 之后紧跟的真实 Y 序列（作为"预测参考基准"）

  区域 C: Test Input (History)   [t=-seq_len .. -1]
           当前测试样本的输入序列 X_test（历史真实）

  区域 D: Pred vs True            [t=0 .. pred_len-1]
           预测值 vs 真实值（未来区间）

X 轴统一为时间偏移（0 为预测起点），所有数据均为原始物理尺度。

布局：GridSpec (rows × cols)，右上角统一 Legend，标题含模型名及关键参数。
"""

import os
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from typing import Optional, Dict, Any, List

# 非交互式后端（兼容无 DISPLAY 环境）
matplotlib.use('Agg')


# ─────────────────────────────────────────────────────────────
# ETT 特征名
# ─────────────────────────────────────────────────────────────

ETT_FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _get_feature_label(feat_idx: int, n_features: int) -> str:
    if feat_idx < len(ETT_FEATURE_NAMES) and n_features == len(ETT_FEATURE_NAMES):
        return ETT_FEATURE_NAMES[feat_idx]
    return "F" + str(feat_idx)


# ─────────────────────────────────────────────────────────────
# 四段数据提取
# ─────────────────────────────────────────────────────────────

def _extract_4segments(
    history: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    seq_len: int,
    pred_len: int,
    n_features: int,
    sample_idx: int,
    feat_idx: int,
    best_match: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    从展平数组中提取四段波形

    Args:
        history: shape (n_samples, seq_len * n_features) — 测试集历史输入（展平）
        preds:   shape (n_samples, pred_len * n_features) — 预测值
        trues:   shape (n_samples, pred_len * n_features) — 真实值
        best_match: 可选，{'hist_match': np.ndarray, 'pred_match': np.ndarray}
                    历史最相似匹配段及对应的真实后续

    Returns:
        dict 含四段数据，均为 shape (N,) 的 1D 数组
    """
    def _get(data, idx, fidx, slen):
        """从展平数据中提取指定样本和特征列"""
        if data.ndim == 1:
            return data
        per_feat = data.shape[1] // n_features
        start = fidx * per_feat
        return data[idx, start:start + slen]

    hist_len = seq_len * n_features // n_features
    # 提取当前样本的特征列（展平为 seq_len）
    seg_a_len = min(seq_len, seq_len)  # 历史回看段长度

    seg_a = _get(history, sample_idx, feat_idx, seq_len)  # 历史回看
    seg_c = _get(history, sample_idx, feat_idx, seq_len)  # 测试输入（与 seg_a 相同数据源）

    seg_d_pred = _get(preds, sample_idx, feat_idx, pred_len)
    seg_d_true = _get(trues, sample_idx, feat_idx, pred_len)

    # 区域 B（历史预测段）：若提供 best_match 则使用，否则用 seg_c 移位近似
    if best_match is not None and 'pred_match' in best_match:
        match_pred = best_match['pred_match']  # shape (pred_len,)
        if match_pred is not None and len(match_pred) >= pred_len:
            seg_b = match_pred[:pred_len]
        else:
            seg_b = np.zeros(pred_len)
    else:
        # fallback：用 seg_a 后 pred_len 个点作为近似（无实际含义，仅作占位）
        seg_b = seg_a[-pred_len:] if len(seg_a) >= pred_len else np.pad(seg_a, (pred_len - len(seg_a), 0), constant_values=0)

    # 区域 A（历史匹配段）
    if best_match is not None and 'hist_match' in best_match:
        match_hist = best_match['hist_match']
        if match_hist is not None and len(match_hist) >= seq_len:
            seg_a = match_hist[:seq_len]
        else:
            seg_a = seg_a
    else:
        seg_a = seg_a  # 直接用测试集历史

    return {
        'seg_a': seg_a.astype(np.float64),   # Historical Lookback
        'seg_b': seg_b.astype(np.float64),   # Historical Prediction
        'seg_c': seg_c.astype(np.float64),   # Test Input
        'seg_d_pred': seg_d_pred.astype(np.float64),
        'seg_d_true': seg_d_true.astype(np.float64),
    }


# ─────────────────────────────────────────────────────────────
# 单图绘制
# ─────────────────────────────────────────────────────────────

def _draw_single_subplot(
    ax: plt.Axes,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    seg_c: np.ndarray,
    seg_d_pred: np.ndarray,
    seg_d_true: np.ndarray,
    seq_len: int,
    pred_len: int,
    model_name: str,
    sample_idx: int,
    feat_label: str,
    params: Dict[str, Any],
    show_regions: bool = True
):
    """
    在一个 ax 上绘制四段波形

    四段 X 轴坐标（统一从左到右时间递增，0 = 预测起点）：
      seg_a: [-seq_len, -1]       Historical Lookback（蓝灰）
      seg_b: [-pred_len, -1]      Historical Prediction（橙黄）
      seg_c: [-seq_len, -1]      Test Input / True History（蓝）
      seg_d: [0, pred_len-1]     Pred vs True（红虚 vs 绿实）
    """
    # X 轴（时间偏移，0 为预测起点）
    x_a = np.arange(-seq_len, 0)                           # [-seq_len, -1]
    x_b = np.arange(-pred_len, 0)                          # [-pred_len, -1]
    x_c = np.arange(-seq_len, 0)                           # [-seq_len, -1]
    x_d = np.arange(0, pred_len)                           # [0, pred_len-1]

    # 区域背景色（可选）
    if show_regions:
        ax.axvspan(-seq_len - 0.5, 0 - 0.5, alpha=0.04, color='gray', zorder=0)

    # 区域 A：Historical Lookback（浅蓝灰）
    ax.plot(x_a, seg_a, color='#7fb3d3', linewidth=1.8, alpha=0.85,
            label='A: Hist. Lookback', zorder=3)

    # 区域 B：Historical Prediction（橙黄）
    if len(x_b) == len(seg_b):
        ax.plot(x_b, seg_b, color='#f5b041', linewidth=1.8, alpha=0.85,
                label='B: Hist. Pred', zorder=3)

    # 区域 C：Test Input（深蓝）
    ax.plot(x_c, seg_c, color='#1f4e79', linewidth=2.2, alpha=0.9,
            label='C: Test Input', zorder=4)

    # 区域 D：Pred vs True
    ax.plot(x_d, seg_d_true, color='#28a745', linewidth=2.2,
            label='D: True (Future)', zorder=5)
    ax.plot(x_d, seg_d_pred, color='#c00000', linewidth=2.2,
            linestyle='--', marker='o', markersize=2.5,
            label='D: Prediction', zorder=6)

    # X=0 分隔线（虚线）
    ax.axvline(x=0, color='#404040', linestyle=':', linewidth=1.5, zorder=2)

    # Y 轴 zeroline
    ax.axhline(y=0, color='lightgray', linewidth=0.8, zorder=1)

    ax.set_xlim(-seq_len - 1, pred_len + 1)
    ax.set_xlabel('时间偏移（0 = 预测起点）', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # 子图标题
    ax.set_title(
        model_name + " | sample=" + str(sample_idx)
        + " | " + feat_label
        + " | pred=" + str(pred_len),
        fontsize=8, pad=3
    )


# ─────────────────────────────────────────────────────────────
# 主绘图函数
# ─────────────────────────────────────────────────────────────

def plot_comparison_samples(
    history: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    seq_len: int,
    pred_len: int,
    n_features: int,
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    n_samples: int = 9,
    figsize: tuple = (14, 10),
    feat_idx: int = -1,
    best_matches: Optional[List[Optional[Dict]]] = None,
    dpi: int = 120
):
    """
    生成四段线对比网格图

    Args:
        history: shape (n, seq_len * n_features) 展平数组，原始物理尺度
        preds:   shape (n, pred_len * n_features)
        trues:   shape (n, pred_len * n_features)
        seq_len, pred_len, n_features: 序列参数
        model_name: 模型名称（用于标题）
        params: 参数字典（用于总图副标题）
        save_path: PNG 保存路径
        n_samples: 网格图子图数量（建议 4, 9, 16）
        figsize: 总图尺寸
        feat_idx: 要展示的特征索引（-1 = 最后一列 Target）
        best_matches: 可选，每个样本的最相似匹配信息列表
        dpi: 图片分辨率
    """
    if feat_idx < 0:
        feat_idx = n_features - 1
    feat_idx = min(feat_idx, n_features - 1)

    # 转为 numpy 数组
    history = np.asarray(history)
    preds = np.asarray(preds)
    trues = np.asarray(trues)

    n_available = min(history.shape[0], preds.shape[0], trues.shape[0])
    if n_available == 0:
        _plot_placeholder(save_path, "无有效数据", figsize)
        return

    n_samples = min(n_samples, n_available)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    feat_label = _get_feature_label(feat_idx, n_features)

    # 随机采样（固定 seed 保证可复现）
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_available, size=n_samples, replace=False).tolist()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.45,
        wspace=0.35
    )

    for i, idx in enumerate(sample_indices):
        r = i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(gs[r, c])

        best_match = None
        if best_matches is not None and i < len(best_matches):
            best_match = best_matches[i]

        segs = _extract_4segments(
            history, preds, trues,
            seq_len, pred_len, n_features,
            sample_idx=idx,
            feat_idx=feat_idx,
            best_match=best_match
        )

        _draw_single_subplot(
            ax,
            segs['seg_a'], segs['seg_b'], segs['seg_c'],
            segs['seg_d_pred'], segs['seg_d_true'],
            seq_len, pred_len,
            model_name, idx, feat_label, params or {}
        )

    # ── 总图标题 ────────────────────────────────────────────
    if params:
        revin_t = params.get('revin_type', 'none')
        topk = params.get('top_k', '-')
        wsize = params.get('word_size', '-')
        title = (
            model_name + " | seq=" + str(seq_len)
            + " pred=" + str(pred_len)
            + " revin=" + revin_t
            + " top_k=" + str(topk)
            + " word_size=" + str(wsize)
            + " | " + feat_label + " | 原始物理尺度"
        )
    else:
        title = model_name + " | seq=" + str(seq_len) + " pred=" + str(pred_len)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)

    # ── 统一 Legend（放在右下角之外）────────────────────────
    handles = [
        Patch(facecolor='#7fb3d3', label='A: Hist. Lookback'),
        Patch(facecolor='#f5b041', label='B: Hist. Prediction'),
        Patch(facecolor='#1f4e79', label='C: Test Input'),
        Patch(facecolor='#28a745', label='D: True (Future)'),
        Patch(facecolor='#c00000', label='D: Prediction (red --)'),
    ]
    fig.legend(
        handles=handles,
        loc='lower right',
        bbox_to_anchor=(0.99, 0.01),
        fontsize=8,
        framealpha=0.9,
        edgecolor='gray',
        ncol=5
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print("[plotting] Saved: " + save_path, flush=True)

    plt.close(fig)


def _plot_placeholder(save_path: Optional[str], message: str, figsize: tuple):
    """当数据无效时生成占位图"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha='center', va='center',
            fontsize=14, transform=ax.transAxes, color='gray')
    ax.axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_summary_bar(
    summary_csv: str,
    metric: str = 'MAE',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
):
    """
    从 summary_metrics.csv 生成对比柱状图

    Args:
        summary_csv: summary_metrics.csv 文件路径
        metric: 展示的指标（MAE / MSE / RMSE / MAPE）
        save_path: PNG 保存路径
    """
    import pandas as pd

    if not os.path.exists(summary_csv):
        print("[plotting] summary_csv not found: " + summary_csv)
        return

    df = pd.read_csv(summary_csv)
    if metric not in df.columns:
        print("[plotting] metric '" + metric + "' not in CSV")
        return

    df_success = df[df['status'] == 'success'].copy()
    if df_success.empty:
        print("[plotting] no successful experiments found")
        return

    # 构造标签：model + revin_type
    df_success['label'] = (
        df_success['model'] + ' (' + df_success['revin_type'].astype(str) + ')'
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_success)))

    bars = ax.bar(
        df_success['label'], df_success[metric],
        color=colors, edgecolor='gray', linewidth=0.5
    )

    for bar, val in zip(bars, df_success[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            '%.4f' % val, ha='center', va='bottom', fontsize=8
        )

    ax.set_title(metric + ' 对比（越低越好）', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xlabel('模型 (归一化类型)', fontsize=10)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print("[plotting] Saved: " + save_path)

    plt.close(fig)


if __name__ == '__main__':
    # 单元测试：生成假数据的占位图
    n = 20
    seq = 96
    pred = 48
    nf = 7

    history_fake = np.random.randn(n, seq * nf).astype(np.float32)
    preds_fake = np.random.randn(n, pred * nf).astype(np.float32) * 0.9
    trues_fake = np.random.randn(n, pred * nf).astype(np.float32)

    out_dir = os.path.join(os.path.dirname(__file__), 'results', '_test_plot')
    os.makedirs(out_dir, exist_ok=True)

    plot_comparison_samples(
        history=history_fake,
        preds=preds_fake,
        trues=trues_fake,
        seq_len=seq,
        pred_len=pred,
        n_features=nf,
        model_name='PatternSearch',
        params={'revin_type': 'dual', 'top_k': 5, 'word_size': 8},
        save_path=os.path.join(out_dir, 'test_visualization.png'),
        n_samples=9,
        feat_idx=nf - 1
    )
    print("Test OK")
