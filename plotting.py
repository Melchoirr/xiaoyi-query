"""
Static Visualization Module - Four-Segment Waveform Comparison

Academic layout (left to right in time order):

  Region A: Historical Lookback    [t=-seq_len .. -1]
             Most similar historical subsequence retrieved by the model
  Region B: Historical Prediction  [t=-pred_len .. -1]
             Ground-truth future following Region A (as reference baseline)
  Region C: Test Input             [t=-seq_len .. -1]
             Current test sample's input sequence
  Region D: Pred vs Ground Truth    [t=0 .. pred_len-1]
             Predicted vs actual future interval

X-axis: time offset (0 = prediction start), Y-axis: original physical scale.
All labels use English-only academic nomenclature for Linux compatibility.
"""

import os
import sys
import json
import numpy as np
from typing import Optional, Dict, Any, List

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

    Args:
        history: (n_samples, seq_len * n_features) — test set history, raw scale
        preds:   (n_samples, pred_len * n_features) — predictions
        trues:   (n_samples, pred_len * n_features) — ground truth
        best_match: optional, {'hist_match': np.ndarray, 'pred_match': np.ndarray}

    Returns:
        dict of 1D arrays for each segment
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
    Draw four-segment waveform on one axes.

    X-axis coordinates (left to right, 0 = prediction start):
      seg_a: [-seq_len, -1]    Historical Lookback (blue-gray)
      seg_b: [-pred_len, -1]   Historical Prediction (orange)
      seg_c: [-seq_len, -1]    Test Input (dark blue)
      seg_d: [0, pred_len-1]   Pred vs True (red dashed vs green solid)
    """
    x_a = np.arange(-seq_len, 0)
    x_b = np.arange(-pred_len, 0)
    x_c = np.arange(-seq_len, 0)
    x_d = np.arange(0, pred_len)

    if show_regions:
        ax.axvspan(-seq_len - 0.5, 0 - 0.5, alpha=0.04, color='gray', zorder=0)

    # Region A: Historical Lookback
    ax.plot(x_a, seg_a, color='#7fb3d3', linewidth=1.8, alpha=0.85,
            label='A: Hist. Lookback', zorder=3)

    # Region B: Historical Prediction
    if len(x_b) == len(seg_b):
        ax.plot(x_b, seg_b, color='#f5b041', linewidth=1.8, alpha=0.85,
                label='B: Hist. Pred', zorder=3)

    # Region C: Test Input
    ax.plot(x_c, seg_c, color='#1f4e79', linewidth=2.2, alpha=0.9,
            label='C: Test Input', zorder=4)

    # Region D: Pred vs True
    ax.plot(x_d, seg_d_true, color='#28a745', linewidth=2.2,
            label='D: Ground Truth', zorder=5)
    ax.plot(x_d, seg_d_pred, color='#c00000', linewidth=2.2,
            linestyle='--', marker='o', markersize=2.5,
            label='D: Prediction', zorder=6)

    # X=0 separator
    ax.axvline(x=0, color='#404040', linestyle=':', linewidth=1.5, zorder=2)
    ax.axhline(y=0, color='lightgray', linewidth=0.8, zorder=1)

    ax.set_xlim(-seq_len - 1, pred_len + 1)
    ax.set_xlabel('Time Offset (0 = Prediction Start)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    ax.set_title(
        model_name + " | sample=" + str(sample_idx)
        + " | " + feat_label
        + " | pred=" + str(pred_len),
        fontsize=8, pad=3
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
    figsize: tuple = (14, 10),
    feat_idx: int = -1,
    best_matches: Optional[List[Optional[Dict]]] = None,
    dpi: int = 120
):
    """
    Generate four-segment waveform comparison grid.

    Args:
        history: (n, seq_len * n_features) flattened, raw physical scale
        preds:   (n, pred_len * n_features)
        trues:   (n, pred_len * n_features)
        seq_len, pred_len, n_features: sequence parameters
        model_name: model name (used in title)
        params:  parameter dict (used in title)
        save_path: PNG save path (default: not saved)
        n_samples: number of subplots (default 9)
        figsize: total figure size (default (14, 10))
        feat_idx: which feature dimension to plot (default -1 = Target)
        best_matches: optional, most-similar-match data (provided by model)
        dpi: PNG resolution (default 120)
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
        _plot_placeholder(save_path, "No valid data", figsize)
        return

    n_samples = min(n_samples, n_available)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    feat_label = _get_feature_label(feat_idx, n_features)

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

    Args:
        summary_csv: path to summary_metrics.csv
        metric: metric to plot (MAE / MSE / RMSE / MAPE / RSE / CORR)
        save_path: PNG save path
        figsize: figure size
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


if __name__ == '__main__':
    # Unit test with fake data
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
