"""
Static Visualization Module - Four-Segment Waveform Comparison + Conference-Level Analysis

v4.0 大道至简重构版新增：
1. plot_super_comparison_matrix: 全局热力图/柱状图网格对比
2. plot_retrieval_fading: 历史匹配溯源图（权重透明度绑定）

Academic layout (left to right in time order):
  Region A: Historical Lookback    [t=-seq_len .. -1]
  Region B: Historical Prediction  [t=-pred_len .. -1]
  Region C: Test Input             [t=-seq_len .. -1]
  Region D: Pred vs Ground Truth    [t=0 .. pred_len-1]

All labels use English-only academic nomenclature for Linux compatibility.
"""

import os
import sys
import json
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

# Soft import matplotlib (may not be installed in venv)
try:
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    _HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    gridspec = None
    Patch = None
    _HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────
# ETT feature names
# ─────────────────────────────────────────────────────────────

ETT_FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _get_feature_label(feat_idx: int, n_features: int) -> str:
    if feat_idx < len(ETT_FEATURE_NAMES) and n_features == len(ETT_FEATURE_NAMES):
        return ETT_FEATURE_NAMES[feat_idx]
    return "F" + str(feat_idx)


# ─────────────────────────────────────────────────────────────
# Four-segment data extraction
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
    Extract four waveform segments from flattened arrays.
    """
    def _get(data, idx, fidx, slen):
        if data.ndim == 1:
            return data
        per_feat = data.shape[1] // n_features
        start = fidx * per_feat
        return data[idx, start:start + slen]

    seg_a = _get(history, sample_idx, feat_idx, seq_len)   # Historical Lookback
    seg_c = _get(history, sample_idx, feat_idx, seq_len)   # Test Input (same source)

    seg_d_pred = _get(preds, sample_idx, feat_idx, pred_len)
    seg_d_true = _get(trues, sample_idx, feat_idx, pred_len)

    # Region B (Historical Prediction)
    if best_match is not None and 'pred_match' in best_match:
        match_pred = best_match['pred_match']
        if match_pred is not None and len(match_pred) >= pred_len:
            seg_b = match_pred[:pred_len]
        else:
            seg_b = np.zeros(pred_len)
    else:
        seg_b = seg_a[-pred_len:] if len(seg_a) >= pred_len \
            else np.pad(seg_a, (pred_len - len(seg_a), 0), constant_values=0)

    # Region A (Historical Match)
    if best_match is not None and 'hist_match' in best_match:
        match_hist = best_match['hist_match']
        if match_hist is not None and len(match_hist) >= seq_len:
            seg_a = match_hist[:seq_len]

    return {
        'seg_a': seg_a.astype(np.float64),
        'seg_b': seg_b.astype(np.float64),
        'seg_c': seg_c.astype(np.float64),
        'seg_d_pred': seg_d_pred.astype(np.float64),
        'seg_d_true': seg_d_true.astype(np.float64),
    }


# ─────────────────────────────────────────────────────────────
# Single subplot drawing
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
    Draw four-segment waveform on one axes (v4.2: 增粗线条，增强对比度).
    """
    x_a = np.arange(-seq_len, 0)
    x_b = np.arange(-pred_len, 0)
    x_c = np.arange(-seq_len, 0)
    x_d = np.arange(0, pred_len)

    if show_regions:
        ax.axvspan(-seq_len - 0.5, 0 - 0.5, alpha=0.06, color='gray', zorder=0)

    # Region A: Historical Lookback (灰色上下文：linewidth=1.5, alpha=0.8)
    ax.plot(x_a, seg_a, color='#7fb3d3', linewidth=1.5, alpha=0.8,
            marker='None', label='A: Hist. Lookback', zorder=3)

    # Region B: Historical Prediction (灰色上下文：linewidth=1.5, alpha=0.8)
    if len(x_b) == len(seg_b):
        ax.plot(x_b, seg_b, color='#f5b041', linewidth=1.5, alpha=0.8,
                marker='None', label='B: Hist. Pred', zorder=3)

    # Region C: Test Input (深蓝：linewidth=1.5, alpha=0.9)
    ax.plot(x_c, seg_c, color='#1f4e79', linewidth=1.5, alpha=0.9,
            marker='None', label='C: Test Input', zorder=4)

    # Region D: Ground Truth (深绿：linewidth=2.0) + Prediction (红色虚线：linewidth=1.5, alpha=0.9)
    ax.plot(x_d, seg_d_true, color='#28a745', linewidth=2.0,
            marker='None', label='D: Ground Truth', zorder=5)
    ax.plot(x_d, seg_d_pred, color='#c00000', linewidth=1.5, alpha=0.9,
            linestyle='--', marker='None', label='D: Prediction', zorder=6)

    # X=0 separator
    ax.axvline(x=0, color='#404040', linestyle=':', linewidth=1.5, zorder=2)
    ax.axhline(y=0, color='lightgray', linewidth=0.8, zorder=1)

    ax.set_xlim(-seq_len - 1, pred_len + 1)
    ax.set_xlabel('Time Offset (0 = Prediction Start)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax.set_title(
        model_name + " | sample=" + str(sample_idx)
        + " | " + feat_label
        + " | pred=" + str(pred_len),
        fontsize=9, pad=4
    )


# ─────────────────────────────────────────────────────────────
# Main plotting function
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
    figsize: tuple = None,  # v4.2: 自动计算最优尺寸
    feat_idx: int = -1,
    best_matches: Optional[List[Optional[Dict]]] = None,
    dpi: int = 300  # v4.2: 固定 300 DPI 高清输出
):
    """
    Generate four-segment waveform comparison grid (v4.2: 优化视觉清晰度).
    """
    if not _HAS_MATPLOTLIB:
        print(
            "[plotting] WARNING: matplotlib not installed. "
            "Install with: pip install matplotlib\n"
            "Skipping plot. Experiments will continue normally."
        )
        return

    if feat_idx < 0:
        feat_idx = n_features - 1
    feat_idx = min(feat_idx, n_features - 1)

    history = np.asarray(history)
    preds = np.asarray(preds)
    trues = np.asarray(trues)

    n_available = min(history.shape[0], preds.shape[0], trues.shape[0])
    if n_available == 0:
        _plot_placeholder(save_path, "No valid data", (10, 6))
        return

    n_samples = min(n_samples, n_available)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    # v4.2: 自动计算最优画布尺寸，确保每个子图有足够的显示空间
    if figsize is None:
        fig_w = max(12, 4 * n_cols)   # 每个子图至少 4 英寸宽
        fig_h = max(8, 2.5 * n_rows)  # 每个子图至少 2.5 英寸高
        figsize = (fig_w, fig_h)

    feat_label = _get_feature_label(feat_idx, n_features)

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_available, size=n_samples, replace=False).tolist()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.5,   # v4.2: 增大垂直间距防止标签重叠
        wspace=0.4    # v4.2: 增大水平间距
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

    # ── Figure title ───────────────────────────────────────────
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
            + " | " + feat_label + " | Raw Physical Scale"
        )
    else:
        title = model_name + " | seq=" + str(seq_len) + " pred=" + str(pred_len)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)

    # ── Unified Legend ────────────────────────────────────────
    handles = [
        Patch(facecolor='#7fb3d3', label='A: Hist. Lookback'),
        Patch(facecolor='#f5b041', label='B: Hist. Prediction'),
        Patch(facecolor='#1f4e79', label='C: Test Input'),
        Patch(facecolor='#28a745', label='D: Ground Truth'),
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
    """Generate placeholder when data is invalid"""
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
    Generate bar chart from summary_metrics.csv (English-only, Linux-safe).
    """
    if not _HAS_MATPLOTLIB:
        print("[plotting] matplotlib not installed, skipping summary bar chart.")
        return

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

    ax.set_title(f'{metric} Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xlabel('Model (RevIN Type)', fontsize=10)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print("[plotting] Saved: " + save_path)

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# v4.0 新增：顶会级分析大图
# ═══════════════════════════════════════════════════════════════

def plot_super_comparison_matrix(
    run_dir: str,
    metric: str = 'MAE',
    save_path: Optional[str] = None,
    figsize: tuple = (20, 12),
    dpi: int = 150
):
    """
    绘制全局对比矩阵热力图/柱状图 - Conference-Level Analysis

    功能：
    1. 读取 run_dir/summary_metrics.csv
    2. 纵轴：7 个算法名称（PatternSearch, RAGSearch 等）
    3. 横轴：不同的超参组合（如 seq96_pred96_k5, seq192_pred96 等）
    4. 颜色深浅代表 MAE/MSE/RMSE 等指标的大小
    5. 支持 Heatmap 和 Grouped Bar Chart 两种模式

    Args:
        run_dir: 实验运行目录
        metric: 要对比的指标（MAE / MSE / RMSE / MAPE）
        save_path: 保存路径（默认 run_dir/super_comparison_matrix.png）
        figsize: 图形大小
        dpi: 分辨率
    """
    if not _HAS_MATPLOTLIB:
        print("[plotting] matplotlib not installed, skipping super comparison matrix.")
        return

    import pandas as pd

    csv_path = os.path.join(run_dir, 'summary_metrics.csv')
    if not os.path.exists(csv_path):
        print(f"[plotting] summary_metrics.csv not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 过滤成功的实验
    df_success = df[df['status'] == 'success'].copy()
    if df_success.empty:
        print("[plotting] No successful experiments found in CSV.")
        return

    if metric not in df_success.columns:
        print(f"[plotting] Metric '{metric}' not found in CSV columns.")
        return

    # 构建超参组合标签
    def make_param_label(row):
        """生成唯一的超参组合标签"""
        parts = [f"s{row['seq_len']}_p{row['pred_len']}"]
        # 根据模型添加特定参数
        if row['model'] in ['PatternSearch', 'DTWSearch', 'MatrixProfileSearch', 'TS2VecSearch']:
            topk = row.get('top_k', 5)
            if pd.notna(topk):
                parts.append(f"k{int(topk)}")
        elif row['model'] == 'LSHSearch':
            h = row.get('n_hash_funcs', 16)
            t = row.get('n_tables', 4)
            if pd.notna(h) and pd.notna(t):
                parts.append(f"h{int(h)}t{int(t)}")
        elif row['model'] == 'SAXSearch':
            w = row.get('word_size', 8)
            a = row.get('alphabet_size', 8)
            if pd.notna(w) and pd.notna(a):
                parts.append(f"w{int(w)}a{int(a)}")
        elif row['model'] == 'RAGSearch':
            dm = row.get('d_model', 256)
            nh = row.get('n_heads', 8)
            if pd.notna(dm) and pd.notna(nh):
                parts.append(f"dm{int(dm)}h{int(nh)}")
        elif row['model'] == 'TS2VecSearch':
            hd = row.get('hidden_dim', 64)
            e = row.get('epochs', 100)
            if pd.notna(hd) and pd.notna(e):
                parts.append(f"hd{int(hd)}e{int(e)}")
        return "_".join(parts)

    df_success['param_label'] = df_success.apply(make_param_label, axis=1)

    # 获取所有模型和参数组合
    models = df_success['model'].unique().tolist()
    param_labels = df_success['param_label'].unique().tolist()

    # 按参数标签长度和字典序排序
    param_labels = sorted(param_labels, key=lambda x: (len(x), x))

    if len(models) == 0 or len(param_labels) == 0:
        print("[plotting] No valid data for comparison matrix.")
        return

    # 创建数据矩阵
    n_models = len(models)
    n_params = len(param_labels)

    # 构建透视表
    pivot = df_success.pivot_table(
        values=metric,
        index='model',
        columns='param_label',
        aggfunc='first'
    )

    # 选择存在的模型
    pivot = pivot.reindex(index=models, columns=param_labels, fill_value=np.nan)

    # 计算均值用于排序
    model_order = pivot.mean(axis=1).sort_values().index.tolist()
    pivot = pivot.reindex(model_order)

    # ── 绘制图形 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.2, 1]})

    # 左图：热力图
    ax1 = axes[0]

    # 过滤掉全为 NaN 的行和列
    valid_cols = pivot.columns[pivot.notna().any()].tolist()
    valid_rows = pivot.index[pivot.notna().any(axis=1)].tolist()

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        print("[plotting] No valid data for heatmap.")
        plt.close(fig)
        return

    data_matrix = pivot.loc[valid_rows, valid_cols].values.astype(float)

    # 绘制热力图
    cmap = 'RdYlGn_r'  # 红=差(大), 绿=好(小)
    im = ax1.imshow(data_matrix, aspect='auto', cmap=cmap, vmin=np.nanmin(data_matrix), vmax=np.nanmax(data_matrix))

    # 设置刻度
    ax1.set_xticks(np.arange(len(valid_cols)))
    ax1.set_yticks(np.arange(len(valid_rows)))
    ax1.set_xticklabels(valid_cols, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(valid_rows, fontsize=9)

    # 添加数值标注
    for i in range(len(valid_rows)):
        for j in range(len(valid_cols)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > (np.nanmax(data_matrix) + np.nanmin(data_matrix)) / 2 else 'black'
                ax1.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=text_color)

    ax1.set_title(f'{metric} Heatmap by Model & Parameters\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Parameter Configuration', fontsize=10)
    ax1.set_ylabel('Model', fontsize=10)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label(metric, fontsize=10)

    # 右图：分组柱状图（Top-10 最佳配置）
    ax2 = axes[1]

    # 获取最佳配置
    df_sorted = df_success.sort_values(metric).head(10)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))
    bars = ax2.barh(
        np.arange(len(df_sorted)),
        df_sorted[metric].values,
        color=colors,
        edgecolor='gray',
        linewidth=0.5
    )

    # 添加数值标签
    for bar, val in zip(bars, df_sorted[metric].values):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    # 设置刻度
    ax2.set_yticks(np.arange(len(df_sorted)))
    labels = [f"{row['model']}\n({row['param_label']})"
              for _, row in df_sorted.iterrows()]
    ax2.set_yticklabels(labels, fontsize=7)

    ax2.set_xlabel(metric, fontsize=10)
    ax2.set_title(f'Top-10 Best Configurations\n(Sorted by {metric})', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()  # 最好的在上面
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

    # 整体标题
    fig.suptitle(
        f'Time-Series Forecasting: Comprehensive Model Comparison\n'
        f'Dataset: {df_success["exp_id"].iloc[0].split("_")[0] if len(df_success) > 0 else "Unknown"} | '
        f'Total Experiments: {len(df_success)} | Metric: {metric}',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is None:
        save_path = os.path.join(run_dir, 'super_comparison_matrix.png')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"[plotting] Super comparison matrix saved: {save_path}")

    plt.close(fig)

    # 同时生成单独的柱状图摘要
    _plot_model_ranking_bar(df_success, metric, run_dir)


def _plot_model_ranking_bar(df_success: 'pd.DataFrame', metric: str, run_dir: str):
    """绘制按模型分组的柱状图排名"""
    if not _HAS_MATPLOTLIB:
        return

    # 计算每个模型的最佳成绩
    model_best = df_success.groupby('model')[metric].min().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))

    n_models = len(model_best)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_models))

    bars = ax.bar(
        range(n_models),
        model_best.values,
        color=colors,
        edgecolor='gray',
        linewidth=1
    )

    # 添加数值标签
    for bar, val in zip(bars, model_best.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_best.index, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(f'Best {metric} (Lower is Better)', fontsize=12)
    ax.set_title(f'Model Ranking by Best {metric}\n(Darker Red = Worse, Green = Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 标注冠军
    best_model = model_best.index[0]
    ax.annotate('CHAMPION', xy=(0, model_best.values[0]),
               xytext=(0.5, model_best.values[0] * 1.1),
               fontsize=10, fontweight='bold', color='green',
               arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()

    save_path = os.path.join(run_dir, 'model_ranking_bar.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[plotting] Model ranking bar saved: {save_path}")
    plt.close(fig)


def plot_retrieval_fading(
    exp_dir: str,
    n_samples: int = 5,
    feat_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: tuple = (18, 12),
    dpi: int = 300  # v4.1: 提高 DPI 确保线条平滑清晰
):
    """
    绘制历史匹配溯源图 - Conference-Level Analysis

    核心视觉：
    1. X 轴分为两段：过去 ([-seq_len, 0]) 和 未来 ([0, pred_len])
    2. 用较粗的实线画出当前样本的 Test Input（过去）和 Ground Truth（未来）
    3. 将匹配到的 K 条历史 topk_histories 和 topk_futures 画在同一张图上
    4. 线条颜色设为绿色，透明度绑定为对应的 topk_weights
    5. 叠加红色的 Final Prediction 虚线

    Args:
        exp_dir: 实验目录（包含 retrieval_meta.npz, preds.npy, trues.npy）
        n_samples: 绘制的样本数量（默认 5）
        feat_idx: 绘制哪个特征维度（默认 -1 = 目标特征）
        save_path: 保存路径
        figsize: 图形大小
        dpi: 分辨率
    """
    if not _HAS_MATPLOTLIB:
        print("[plotting] matplotlib not installed, skipping retrieval fading plot.")
        return

    # 加载数据
    meta_path = os.path.join(exp_dir, 'retrieval_meta.npz')
    preds_path = os.path.join(exp_dir, 'preds.npy')
    trues_path = os.path.join(exp_dir, 'trues.npy')
    x_test_path = os.path.join(exp_dir, 'X_test.npy')

    if not os.path.exists(meta_path):
        print(f"[plotting] retrieval_meta.npz not found: {meta_path}")
        print("[plotting] This feature requires retrieval models (PatternSearch, DTWSearch, TS2VecSearch).")
        return

    try:
        meta = np.load(meta_path, allow_pickle=True)
        preds = np.load(preds_path)
        trues = np.load(trues_path)
        x_test = np.load(x_test_path)
    except Exception as e:
        print(f"[plotting] Failed to load data: {e}")
        return

    # 提取元数据
    topk_histories = meta['topk_histories']  # (n_samples, k, seq_len, n_feat)
    topk_futures = meta['topk_futures']      # (n_samples, k, pred_len, n_feat)
    topk_weights = meta['topk_weights']       # (n_samples, k)
    seq_len = int(meta.get('seq_len', 96))
    pred_len = int(meta.get('pred_len', 96))
    n_features = int(meta.get('n_features', 7))
    model_name = str(meta.get('model_name', 'Unknown'))

    if feat_idx < 0:
        feat_idx = n_features - 1
    feat_idx = min(feat_idx, n_features - 1)

    feat_label = _get_feature_label(feat_idx, n_features)

    # 限制样本数量
    n_samples = min(n_samples, topk_histories.shape[0], preds.shape[0])

    # 调整 preds 和 trues 形状
    if preds.ndim == 3:
        preds = preds[:, :, feat_idx]  # (n_test, pred_len)
    if trues.ndim == 3:
        trues = trues[:, :, feat_idx]  # (n_test, pred_len)

    # 调整 x_test 形状
    if x_test.ndim == 2:
        per_feat = x_test.shape[1] // n_features
        x_test_feat = x_test[:, feat_idx * per_feat:(feat_idx + 1) * per_feat]
        if per_feat > seq_len:
            x_test_feat = x_test_feat[:, :seq_len]
        elif per_feat < seq_len:
            x_test_feat = np.pad(x_test_feat, ((0, 0), (0, seq_len - per_feat)), mode='edge')
    else:
        x_test_feat = x_test[:, :seq_len]

    # 颜色映射（绿色系，权重越高越深）
    base_color = '#2ecc71'  # 基础绿色

    # 创建图形
    n_cols = 1
    n_rows = n_samples

    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_samples / 3))
    gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.4)

    for i in range(n_samples):
        ax = fig.add_subplot(gs[i, 0])

        k = topk_histories.shape[1]  # top_k 值

        # X 轴坐标
        x_past = np.arange(-seq_len, 0)
        x_future = np.arange(0, pred_len)
        x_total = np.arange(-seq_len, pred_len)

        # ── 绘制匹配的历史序列（绿色，透明度=权重）──────────────
        for j in range(k):
            weight = topk_weights[i, j]
            alpha = 0.15 + 0.7 * weight  # 权重 0 时 alpha=0.15, 权重 1 时 alpha=0.85

            # 绘制历史匹配（去雾化）
            hist_seq = topk_histories[i, j, :, feat_idx] if topk_histories.ndim == 4 else topk_histories[i, j, :]
            ax.plot(x_past, hist_seq, color=base_color, alpha=alpha, linewidth=1.2,
                   marker='None', label=f'Match {j+1} (w={weight:.2f})' if i == 0 else '')

            # 绘制未来匹配（去雾化）
            if topk_futures.ndim == 4:
                fut_seq = topk_futures[i, j, :, feat_idx]
            else:
                fut_seq = topk_futures[i, j, :]
            ax.plot(x_future, fut_seq, color=base_color, alpha=alpha, linewidth=1.2,
                   marker='None')

        # ── 绘制当前样本的 Test Input（深蓝色粗实线，去雾化）──────
        test_input = x_test_feat[i] if x_test_feat.ndim == 1 else x_test_feat[i, :seq_len]
        ax.plot(x_past, test_input, color='#1f4e79', linewidth=2.5, alpha=0.9,
               marker='None', label='Test Input', zorder=5)

        # ── 绘制 Ground Truth（深绿色粗实线，去雾化）────────────
        gt_future = trues[i] if trues.ndim == 1 else trues[i, :]
        ax.plot(x_future, gt_future, color='#27ae60', linewidth=2.5, alpha=0.9,
               marker='None', label='Ground Truth', zorder=6)

        # ── 绘制 Final Prediction（红色虚线，去雾化）─────────────
        pred_future = preds[i] if preds.ndim == 1 else preds[i, :]
        ax.plot(x_future, pred_future, color='#c0392b', linewidth=2.5, alpha=0.9,
               linestyle='--', marker='None', label='Prediction', zorder=7)

        # ── X=0 分隔线 ────────────────────────────────────────────
        ax.axvline(x=0, color='#404040', linestyle=':', linewidth=1.5, zorder=2)

        # ── 区域填充 ──────────────────────────────────────────────
        ax.axvspan(-seq_len - 0.5, -0.5, alpha=0.05, color='blue', zorder=0)
        ax.axvspan(-0.5, pred_len - 0.5, alpha=0.05, color='green', zorder=0)

        # ── 设置坐标轴 ────────────────────────────────────────────
        ax.set_xlim(-seq_len - 1, pred_len + 1)
        ax.set_xlabel('Time Offset (0 = Prediction Start)', fontsize=10)
        ax.set_ylabel(feat_label, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # ── 标题 ──────────────────────────────────────────────────
        best_idx = np.argmax(topk_weights[i])
        best_weight = topk_weights[i, best_idx]
        ax.set_title(
            f'Sample {i} | Model: {model_name} | '
            f'Top-{k} Retrieval | Best Match: #{best_idx+1} (w={best_weight:.3f})',
            fontsize=11, fontweight='bold'
        )

        # ── 图例（只在第一个子图显示）─────────────────────────────
        if i == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # ── 整体标题 ────────────────────────────────────────────────
    fig.suptitle(
        f'Retrieval Evidence Analysis: {model_name}\n'
        f'Top-K Historical Matches with Weight-Based Transparency\n'
        f'(Greener = Higher Weight, More Transparent = Lower Weight)',
        fontsize=14, fontweight='bold', y=0.99
    )

    # 添加说明文字
    fig.text(0.5, 0.01,
            'Green Lines: Historical Matches (alpha = attention weight) | '
            'Blue Line: Test Input | '
            'Green Solid: Ground Truth | '
            'Red Dashed: Prediction',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path is None:
        save_path = os.path.join(exp_dir, 'retrieval_analysis.png')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"[plotting] Retrieval fading plot saved: {save_path}")

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 批量绘图工具（供 Shell 脚本调用）
# ═══════════════════════════════════════════════════════════════

def generate_all_plots(run_dir: str):
    """
    生成所有分析图表

    调用方式：
        python -c "from plotting import generate_all_plots; generate_all_plots('./results/run_001')"

    Args:
        run_dir: 实验运行目录
    """
    print(f"[plotting] Generating all plots for: {run_dir}")

    # 1. 生成超级对比矩阵
    try:
        plot_super_comparison_matrix(run_dir, metric='MAE')
    except Exception as e:
        print(f"[plotting] Failed to generate super comparison matrix: {e}")

    # 2. 生成模型排名柱状图
    try:
        import pandas as pd
        csv_path = os.path.join(run_dir, 'summary_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_success = df[df['status'] == 'success']
            _plot_model_ranking_bar(df_success, 'MAE', run_dir)
    except Exception as e:
        print(f"[plotting] Failed to generate model ranking: {e}")

    # 3. 遍历所有实验目录，生成溯源图
    try:
        for exp_id in os.listdir(run_dir):
            exp_path = os.path.join(run_dir, exp_id)
            if os.path.isdir(exp_path) and os.path.exists(os.path.join(exp_path, 'retrieval_meta.npz')):
                try:
                    plot_retrieval_fading(exp_path, n_samples=5)
                except Exception as e:
                    print(f"[plotting] Failed to generate retrieval plot for {exp_id}: {e}")
    except Exception as e:
        print(f"[plotting] Failed to scan experiment directories: {e}")

    # 4. 生成顶会级多通道对比网格图 (v4.3)
    try:
        plot_cross_model_comparison(
            run_dir,
            sample_indices=None,  # 默认 [0, N//2, N-1]
            dpi=300
        )
        print("[plotting] Paper-level comparison plot generated!")
    except Exception as e:
        print(f"[plotting] Failed to generate paper-level comparison: {e}")

    print("[plotting] All plots generation completed.")


def plot_cross_model_comparison(
    run_dir: str,
    sample_indices: list = None,
    feat_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: tuple = None,
    dpi: int = 300
):
    """
    顶会级多通道对比网格图 (v4.3: 重写)

    布局：
    - 行数 = 通道数（n_features，例如 7）
    - 列数 = 代表性样本数（默认 [0, N_test // 2, N_test - 1]）
    - figsize = (6 * n_cols, 2.5 * n_features)

    配色：
    - 历史输入 (X_test): #aaaaaa, linewidth=0.8, alpha=0.7
    - 真实值 (Ground Truth): #333333, linewidth=0.5
    - 各个模型 (preds): matplotlib 默认颜色循环, linewidth=0.3, alpha=0.85
    - x=seq_len 处画 #cccccc 垂直虚线分割过去与未来

    保存：run_dir/paper_level_comparison.png, dpi=300, bbox_inches='tight'

    Args:
        run_dir: 实验运行目录
        sample_indices: 采样的测试样本索引列表，默认 [0, N//2, N-1]
        feat_idx: 绘制哪个特征维度（默认 -1 = 所有通道）
        save_path: 保存路径
        figsize: 图形大小
        dpi: 分辨率
    """
    if not _HAS_MATPLOTLIB:
        print("[plotting] matplotlib not installed, skipping paper-level comparison.")
        return

    # ── 1. 收集所有实验目录 ────────────────────────────────────────
    exp_dirs = []
    for item in os.listdir(run_dir):
        exp_path = os.path.join(run_dir, item)
        if not os.path.isdir(exp_path):
            continue
        preds_path = os.path.join(exp_path, 'preds.npy')
        if os.path.exists(preds_path):
            exp_dirs.append({
                'path': exp_path,
                'exp_id': item,
                'preds_path': preds_path,
            })

    if len(exp_dirs) == 0:
        print(f"[plotting] No experiment directories found in {run_dir}")
        return

    # ── 2. 加载共享的参考数据 ──────────────────────────────────────
    ref_exp = exp_dirs[0]
    x_test_path = os.path.join(ref_exp['path'], 'X_test.npy')
    trues_path = os.path.join(ref_exp['path'], 'trues.npy')
    params_path = os.path.join(ref_exp['path'], 'params.json')

    if not (os.path.exists(x_test_path) and os.path.exists(trues_path)):
        print(f"[plotting] Reference data (X_test.npy/trues.npy) not found.")
        return

    try:
        X_test = np.load(x_test_path)      # shape: (N_test, seq_len * n_feat) 或 (N_test, seq_len, n_feat)
        trues = np.load(trues_path)         # shape: (N_test, pred_len * n_feat) 或 (N_test, pred_len, n_feat)
        seq_len = 96
        pred_len = trues.shape[1] if trues.ndim == 2 else trues.shape[1]
        n_features = 7
    except Exception as e:
        print(f"[plotting] Failed to load reference data: {e}")
        return

    # 从 params.json 读取配置
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                p = json.load(f)
                seq_len = p.get('seq_len', seq_len)
                pred_len = p.get('pred_len', pred_len)
                n_features = p.get('n_features', n_features)
        except Exception:
            pass

    # ── 3. 加载所有模型的预测 ────────────────────────────────────
    model_preds = []
    model_names = []

    for exp in exp_dirs:
        try:
            preds = np.load(exp['preds_path'])
            p_path = os.path.join(exp['path'], 'params.json')
            if os.path.exists(p_path):
                with open(p_path, 'r') as f:
                    pj = json.load(f)
                    model_name = pj.get('model_name', exp['exp_id'])
            else:
                model_name = exp['exp_id']
            model_preds.append(preds)
            model_names.append(model_name)
        except Exception as e:
            print(f"[plotting] Failed to load {exp['path']}: {e}")

    if not model_preds:
        print("[plotting] No valid model predictions loaded.")
        return

    # ── 4. 确定采样样本 ──────────────────────────────────────────
    N_test = X_test.shape[0]
    if sample_indices is None:
        sample_indices = [0, N_test // 2, N_test - 1]
    else:
        sample_indices = [min(i, N_test - 1) for i in sample_indices]
    n_cols = len(sample_indices)

    # ── 5. 自动计算画布尺寸 ──────────────────────────────────────
    if figsize is None:
        fig_w = 6 * n_cols
        fig_h = 2.5 * n_features
        figsize = (fig_w, fig_h)

    # ── 6. 处理多通道数据 ─────────────────────────────────────────
    # X_test 和 trues 可能是 (N, seq_len*n_feat) 或 (N, seq_len, n_feat)
    if X_test.ndim == 2:
        per_feat = X_test.shape[1] // n_features
        X_test_3d = X_test.reshape(N_test, seq_len, n_features) if per_feat == seq_len else X_test.reshape(N_test, per_feat, n_features)
        if X_test_3d.shape[1] != seq_len:
            X_test_3d = X_test.reshape(N_test, seq_len, n_features)
    else:
        X_test_3d = X_test

    if trues.ndim == 2:
        per_feat = trues.shape[1] // n_features
        trues_3d = trues.reshape(N_test, pred_len, n_features) if per_feat == pred_len else trues.reshape(N_test, per_feat, n_features)
        if trues_3d.shape[1] != pred_len:
            trues_3d = trues.reshape(N_test, pred_len, n_features)
    else:
        trues_3d = trues

    # 处理 preds 维度
    preds_3d_list = []
    for preds in model_preds:
        if preds.ndim == 2:
            per_feat = preds.shape[1] // n_features
            p3d = preds.reshape(N_test, pred_len, n_features) if per_feat == pred_len else preds.reshape(N_test, per_feat, n_features)
            if p3d.shape[1] != pred_len:
                p3d = preds.reshape(N_test, pred_len, n_features)
            preds_3d_list.append(p3d)
        else:
            preds_3d_list.append(preds)

    # ── 7. 绘制网格图 ─────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=n_features,
        ncols=n_cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=False
    )

    # matplotlib 默认颜色循环（高对比度）
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']

    for row_feat in range(n_features):
        feat_label = _get_feature_label(row_feat, n_features)

        for col_sample, sample_id in enumerate(sample_indices):
            ax = axes[row_feat, col_sample]

            # ── X_test 历史（#aaaaaa, linewidth=0.8, alpha=0.7）────────────
            test_input = X_test_3d[sample_id, :, row_feat] if X_test_3d.ndim == 3 else X_test_3d[sample_id]
            ax.plot(
                np.arange(-seq_len, 0),
                test_input,
                color='#aaaaaa', linewidth=0.8, alpha=0.7,
                zorder=1
            )

            # ── Ground Truth（#333333, linewidth=0.5）──────────────────
            gt_future = trues_3d[sample_id, :, row_feat] if trues_3d.ndim == 3 else trues_3d[sample_id]
            ax.plot(
                np.arange(0, pred_len),
                gt_future,
                color='#333333', linewidth=0.5,
                zorder=2
            )

            # ── 各模型预测（默认颜色循环, linewidth=0.3, alpha=0.85）────
            for model_idx, (p3d, model_name) in enumerate(zip(preds_3d_list, model_names)):
                pred_fut = p3d[sample_id, :, row_feat] if p3d.ndim == 3 else p3d[sample_id]
                color = default_colors[model_idx % len(default_colors)]
                ax.plot(
                    np.arange(0, pred_len),
                    pred_fut,
                    color=color, linewidth=0.3, alpha=0.85,
                    zorder=3 + model_idx
                )

            # ── x = seq_len 垂直虚线分割线 ─────────────────────────
            ax.axvline(
                x=0,
                color='#cccccc', linestyle='--', linewidth=0.8,
                zorder=0
            )

            # ── 子图装饰 ───────────────────────────────────────────
            if col_sample == 0:
                ax.set_ylabel(feat_label, fontsize=6)
            if row_feat == 0:
                ax.set_title(f'Sample {sample_id}', fontsize=7, fontweight='bold')
            if row_feat == n_features - 1:
                ax.set_xlabel('Time', fontsize=6)

            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.3)

    # ── 统一图例 ─────────────────────────────────────────────────
    handles = [
        Patch(facecolor='#aaaaaa', label='X_test (History)'),
        Patch(facecolor='#333333', label='Ground Truth'),
    ]
    for mi, mn in enumerate(model_names):
        short = mn.replace('Search', '').replace('_', ' ')
        color = default_colors[mi % len(default_colors)]
        handles.append(Patch(facecolor=color, label=short))

    fig.legend(
        handles=handles,
        loc='lower right',
        bbox_to_anchor=(0.99, 0.01),
        fontsize=7,
        framealpha=0.9,
        edgecolor='gray',
        ncol=min(4, len(handles)),
    )

    # ── 整体标题 ────────────────────────────────────────────────
    fig.suptitle(
        'Cross-Model Multi-Channel Comparison\n'
        f'Samples: {sample_indices} | Physical Scale',
        fontsize=11, fontweight='bold', y=0.99
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    # ── 保存 ────────────────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(run_dir, 'paper_level_comparison.png')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"[plotting] Paper-level comparison saved: {save_path}")

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 单元测试
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 基础单元测试
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

    # 测试溯源图（生成假数据）
    meta_fake = {
        'topk_histories': np.random.randn(5, 5, 96, 7).astype(np.float32),
        'topk_futures': np.random.randn(5, 5, 48, 7).astype(np.float32),
        'topk_weights': np.random.rand(5, 5).astype(np.float32),
        'topk_weights': np.abs(np.random.rand(5, 5).astype(np.float32)),
    }
    # 归一化权重
    meta_fake['topk_weights'] = meta_fake['topk_weights'] / meta_fake['topk_weights'].sum(axis=1, keepdims=True)
    meta_fake['seq_len'] = 96
    meta_fake['pred_len'] = 48
    meta_fake['n_features'] = 7
    meta_fake['model_name'] = 'PatternSearch'

    np.savez(os.path.join(out_dir, 'retrieval_meta.npz'), **meta_fake)
    np.save(os.path.join(out_dir, 'preds.npy'), preds_fake[:10])
    np.save(os.path.join(out_dir, 'trues.npy'), trues_fake[:10])
    np.save(os.path.join(out_dir, 'X_test.npy'), history_fake[:10])

    try:
        plot_retrieval_fading(out_dir, n_samples=3, save_path=os.path.join(out_dir, 'test_retrieval.png'))
    except Exception as e:
        print(f"Retrieval plot test skipped: {e}")

    print("All unit tests completed.")
