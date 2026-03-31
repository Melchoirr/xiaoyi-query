"""
run.py - 时序预测基线系统 (v4.0 大道至简重构版)

核心设计理念：Python 端纯粹执行器，Shell 端智能调度

变更记录：
- v4.0: 删除所有参数网格展开逻辑（ExperimentRunner, _expand_configs）
-          argparse 简化为标量输入，单次调用仅执行单一实验
-          结果追加写入 summary_metrics.csv，Shell 端全权负责并行调度
-          检索模型新增溯源证据输出（retrieval_meta.npz）
"""

import os
import sys
import gc
import json
import time
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# ============================================================
# 全局配置
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 日志配置
# ============================================================

def _setup_logger(run_dir: str, exp_id: str) -> logging.Logger:
    """为每个实验配置独立日志器"""
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log = logging.getLogger(f"run_{exp_id}")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(os.path.join(log_dir, f"{exp_id}.log"), encoding='utf-8')
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY = {
    'PatternSearch': {
        'class': None,
        'params': ['top_k', 'weighted', 'predict_chunk_size']
    },
    'LSHSearch': {
        'class': None,
        'params': ['n_hash_funcs', 'n_tables', 'hamming_radius',
                   'candidate_cap_per_table', 'candidate_cap_total', 'weighted']
    },
    'SAXSearch': {
        'class': None,
        'params': ['word_size', 'alphabet_size', 'epsilon_threshold',
                   'bucket_top_k', 'weighted']
    },
    'DTWSearch': {
        'class': None,
        'params': ['top_k', 'dtw_radius', 'weighted', 'predict_chunk_size']
    },
    'MatrixProfileSearch': {
        'class': None,
        'params': ['top_k', 'subsequence_length', 'normalize',
                   'predict_chunk_size', 'train_chunk_size']
    },
    'TS2VecSearch': {
        'class': None,
        'params': ['hidden_dim', 'epochs', 'batch_size', 'top_k',
                   'lr', 'temperature']
    },
    'RAGSearch': {
        'class': None,
        'params': ['d_model', 'n_heads', 'epochs', 'batch_size',
                   'lr', 'weight_decay']
    },
}


def import_models():
    """延迟导入所有模型类"""
    from models.PatternSearch import PatternSearch
    from models.LSHSearch import LSHSearch
    from models.SAXSearch import SAXSearch
    from models.DTWSearch import DTWSearch
    from models.MatrixProfileSearch import MatrixProfileSearch
    from models.TS2VecSearch import TS2VecSearch
    from models.RAGSearch import RAGSearch

    MODEL_REGISTRY['PatternSearch']['class'] = PatternSearch
    MODEL_REGISTRY['LSHSearch']['class'] = LSHSearch
    MODEL_REGISTRY['SAXSearch']['class'] = SAXSearch
    MODEL_REGISTRY['DTWSearch']['class'] = DTWSearch
    MODEL_REGISTRY['MatrixProfileSearch']['class'] = MatrixProfileSearch
    MODEL_REGISTRY['TS2VecSearch']['class'] = TS2VecSearch
    MODEL_REGISTRY['RAGSearch']['class'] = RAGSearch


# ============================================================
# Dual-Dimension RevIN 工具函数
# ============================================================

def _safe_std(std: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """数值安全：将 std < threshold 的位置强制置 1.0，防止除零"""
    return np.where(std < threshold, 1.0, std)


def apply_revin(
    X: np.ndarray,
    revin_type: str,
    prefix: str = 'X'
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Dual-Dimension RevIN 归一化 (v4.1 修复)

    核心修复：将 axis=-1 改为 axis=(1, 2)，确保统计量形状为 (B, 1, 1)
    可以完美广播到任意 (B, T, F) 形状的张量（无论 seq_len == pred_len 或不相等）

    Args:
        X: shape (n, seq_len, n_feat) 的 float32 数组
        revin_type: 'none' | 'temporal' | 'feature' | 'dual'
        prefix: 日志前缀

    Returns:
        X_norm: 归一化后的数组
        revin_stats: 字典，key 含 'mean/feat', 'std/feat', 'mean/temp', 'std/temp'
    """
    n, seq, n_feat = X.shape
    stats = {}
    X_out = X.astype(np.float32)

    if revin_type == 'none':
        return X_out, stats

    elif revin_type == 'temporal':
        mean = np.mean(X, axis=1, keepdims=True)
        std = _safe_std(np.std(X, axis=1, keepdims=True))
        X_out = ((X - mean) / std).astype(np.float32)
        stats = {'mean_t': mean, 'std_t': std}
        logger.info(f"[RevIN-{prefix}] temporal: mean={mean.shape}, std={std.shape}")

    elif revin_type == 'feature':
        # 修复：同时聚合时间步和特征维度，得到 (B, 1, 1) 可广播标量
        mean = np.mean(X, axis=(1, 2), keepdims=True)  # (B, 1, 1)
        std = _safe_std(np.std(X, axis=(1, 2), keepdims=True))  # (B, 1, 1)
        X_out = ((X - mean) / std).astype(np.float32)
        stats = {'mean_f': mean, 'std_f': std}
        logger.info(f"[RevIN-{prefix}] feature: mean={mean.shape}, std={std.shape}")

    elif revin_type == 'dual':
        # 修复：feature 归一化先聚合 (B, T, F) → (B, 1, 1)，可广播到任意形状
        mean_f = np.mean(X, axis=(1, 2), keepdims=True)  # (B, 1, 1)
        std_f = _safe_std(np.std(X, axis=(1, 2), keepdims=True))  # (B, 1, 1)
        X_f = ((X - mean_f) / std_f).astype(np.float32)
        mean_t = np.mean(X_f, axis=1, keepdims=True)  # (B, 1, F)
        std_t = _safe_std(np.std(X_f, axis=1, keepdims=True))  # (B, 1, F)
        X_out = ((X_f - mean_t) / std_t).astype(np.float32)
        stats = {'mean_f': mean_f, 'std_f': std_f, 'mean_t': mean_t, 'std_t': std_t}
        logger.info(f"[RevIN-{prefix}] dual: feature {mean_f.shape} -> temporal {mean_t.shape}")

    else:
        logger.warning(f"[RevIN-{prefix}] unknown revin_type='{revin_type}', skipped")

    return X_out, stats


def inverse_revin(
    Y_norm: np.ndarray,
    revin_stats: Dict[str, np.ndarray],
    revin_type: str
) -> np.ndarray:
    """
    Dual-Dimension RevIN 反归一化 (v4.1 修复)

    核心修复：支持任意形状 (B, T, F) 的输入，无论是 seq_len == pred_len 或不相等
    """
    if revin_type == 'none':
        return Y_norm.astype(np.float32)

    elif revin_type == 'temporal':
        mean_t = revin_stats['mean_t']
        std_t = revin_stats['std_t']
        return (Y_norm * std_t + mean_t).astype(np.float32)

    elif revin_type == 'feature':
        mean_f = revin_stats['mean_f']
        std_f = revin_stats['std_f']
        # mean_f/std_f 形状为 (B, 1, 1)，可广播到任意 (B, T, F)
        return (Y_norm * std_f + mean_f).astype(np.float32)

    elif revin_type == 'dual':
        mean_t = revin_stats['mean_t']
        std_t = revin_stats['std_t']
        mean_f = revin_stats['mean_f']
        std_f = revin_stats['std_f']
        # 第一步：reverse temporal normalization
        Y_t = Y_norm * std_t + mean_t  # (B, T, F)
        # 第二步：reverse feature normalization (mean_f/std_f 形状 (B, 1, 1))
        Y_out = Y_t * std_f + mean_f
        return Y_out.astype(np.float32)

    return Y_norm.astype(np.float32)


# ============================================================
# 核心计算逻辑（v4.0: 单次实验，输出溯源证据）
# ============================================================

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单次实验 (v4.0)

    设计变更：
    1. 每个 Python 进程仅执行单一实验
    2. 检索模型输出溯源证据（retrieval_meta.npz）
    3. 结果追加到 summary_metrics.csv
    """
    import numpy as np
    from data_provider.data_loader import get_data, get_X_Y_from_dataset
    from utils.metrics import calculate_all_metrics

    start_time = time.time()
    model_name = config['model_name']
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    revin_type = config.get('revin_type', 'none')
    run_dir = config['run_dir']

    # 生成 exp_id
    exp_id = _make_exp_id(model_name, seq_len, pred_len, config)
    exp_dir = os.path.join(run_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # 实验级日志器
    log = _setup_logger(run_dir, exp_id)
    log.info("=" * 60)
    log.info(f"Experiment {exp_id} started")
    log.info(f"RevIN: {revin_type}")
    log.info("=" * 60)

    try:
        import_models()
        ModelClass = MODEL_REGISTRY[model_name]['class']

        model_params = {}
        for param in MODEL_REGISTRY[model_name]['params']:
            model_params[param] = config.get(param, _get_default(param))

        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if config.get('use_gpu', False) and device == 'cpu':
            log.warning(f"[{model_name}] GPU requested but unavailable, fallback to CPU")
        model_params['device'] = device
        log.info(f"[{model_name}] device={device}, revin_type={revin_type}")

        # 数据加载
        class Args:
            pass
        args = Args()
        args.root_path = config.get('root_path', './ETT_data')
        args.data_path = config.get('data_path', 'ETTm1.csv')
        args.seq_len = seq_len
        args.pred_len = pred_len
        args.features = config.get('features', 'M')
        args.target = config.get('target', 'OT')

        log.info(f"[{model_name}] Loading data seq={seq_len} pred={pred_len}")
        train_set, val_set, test_set = get_data(args)
        X_train, Y_train = get_X_Y_from_dataset(train_set)
        X_test, Y_test = get_X_Y_from_dataset(test_set)

        log.info(f"[{model_name}] Raw: X_train={X_train.shape}, Y_train={Y_train.shape}, "
                 f"X_test={X_test.shape}, Y_test={Y_test.shape}")

        # Y 截断
        if Y_train.ndim == 3:
            Y_train = Y_train[:, -pred_len:, :]
        elif Y_train.ndim == 2:
            Y_train = Y_train[:, -pred_len:]

        if Y_test.ndim == 3:
            Y_test = Y_test[:, -pred_len:, :]
        elif Y_test.ndim == 2:
            Y_test = Y_test[:, -pred_len:]

        log.info(f"[{model_name}] Y truncated: Y_train={Y_train.shape}, Y_test={Y_test.shape}")

        # 3D 化
        X_train_3d = X_train.reshape(X_train.shape[0], seq_len, -1) \
            if X_train.ndim == 2 else X_train.astype(np.float32)
        Y_train_3d = Y_train.reshape(Y_train.shape[0], pred_len, -1) \
            if Y_train.ndim == 2 else Y_train.astype(np.float32)

        n_feat = X_train_3d.shape[-1]
        MAX_PREVIEW = 100
        history_preview_raw = X_test[:MAX_PREVIEW].copy()
        log.info(f"[{model_name}] Saving {MAX_PREVIEW} history previews")

        del train_set, val_set
        gc.collect()

        # Dual-Dimension RevIN 训练阶段
        X_train_norm, revin_stats_train = apply_revin(
            X_train_3d, revin_type, prefix=model_name + '_train'
        )
        Y_train_norm, _ = apply_revin(
            Y_train_3d, revin_type, prefix=model_name + '_train_Y'
        )
        log.info(f"[{model_name}] Training data normalized")

        del X_train, Y_train
        gc.collect()

        # 模型训练
        log.info(f"[{model_name}] Training...")
        model = ModelClass(**model_params)
        model.fit(X_train_norm, Y_train_norm)

        del X_train_3d, Y_train_3d, X_train_norm, Y_train_norm
        gc.collect()

        # 推理阶段
        X_test_3d = X_test.reshape(X_test.shape[0], seq_len, -1) \
            if X_test.ndim == 2 else X_test.astype(np.float32)
        Y_test_3d = Y_test.reshape(Y_test.shape[0], pred_len, -1) \
            if Y_test.ndim == 2 else Y_test.astype(np.float32)

        X_test_norm, revin_stats_test = apply_revin(
            X_test_3d, revin_type, prefix=model_name + '_test'
        )
        log.info(f"[{model_name}] Test data normalized")

        del X_test
        gc.collect()

        log.info(f"[{model_name}] Predicting {X_test_3d.shape[0]} samples...")

        # 检查模型是否支持溯源证据输出
        retrieval_meta = None
        if hasattr(model, 'get_retrieval_meta'):
            log.info(f"[{model_name}] Model supports retrieval evidence, capturing metadata...")
            Y_pred_norm, retrieval_meta = model.get_retrieval_meta(X_test_norm)
            # 先保存归一化版本的 npz
            if retrieval_meta is not None:
                np.savez_compressed(os.path.join(exp_dir, 'retrieval_meta.npz'), **retrieval_meta)
                log.info(f"[{model_name}] Raw retrieval meta saved: hist={retrieval_meta.get('topk_histories', np.array([])).shape}")
        else:
            Y_pred_norm = model.predict(X_test_norm)

        del model, X_test_norm
        gc.collect()

        # ═══════════════════════════════════════════════════════════════════
        # v4.3 修复：指标必须在正确空间计算，X_test 必须反归一化
        # ═══════════════════════════════════════════════════════════════════

        # 1. 先执行 RevIN 反归一化，回到 TSLib 空间
        Y_pred_tslib = inverse_revin(Y_pred_norm, revin_stats_test, revin_type)
        del Y_pred_norm
        gc.collect()

        # 2. 在正确的 TSLib 空间下计算 Metrics（不是归一化空间！）
        metrics = calculate_all_metrics(Y_pred_tslib, Y_test_3d)

        elapsed = time.time() - start_time
        log.info(f"[{model_name}] TSLib-space MAE={metrics.get('MAE', 0):.4f} "
                 f"MSE={metrics.get('MSE', 0):.4f} elapsed={elapsed:.1f}s")

        # 3. 全局反标准化，恢复真实物理尺度（必须在 del test_set 之前）
        n_test, p_len, n_feat = Y_pred_tslib.shape
        Y_pred_flat = Y_pred_tslib.reshape(-1, n_feat)
        Y_test_flat = Y_test_3d.reshape(-1, n_feat)

        Y_pred_inv = test_set.inverse_transform(Y_pred_flat).reshape(n_test, p_len, n_feat).astype(np.float32)
        Y_test_inv = test_set.inverse_transform(Y_test_flat).reshape(n_test, p_len, n_feat).astype(np.float32)

        # 4. 修复 X_Test 的量纲：也执行物理尺度恢复！
        X_test_flat_phys = history_preview_raw.reshape(-1, n_feat)
        X_test_inv_phys = test_set.inverse_transform(X_test_flat_phys).reshape(MAX_PREVIEW, seq_len, n_feat).astype(np.float32)

        # 5. 保存 npy（全部都是物理尺度）
        np.save(os.path.join(exp_dir, 'preds.npy'), Y_pred_inv)
        np.save(os.path.join(exp_dir, 'trues.npy'), Y_test_inv)
        np.save(os.path.join(exp_dir, 'X_test.npy'), X_test_inv_phys.reshape(MAX_PREVIEW, -1))

        # 6. 现在可以安全删除 test_set
        del test_set, Y_pred_flat, Y_test_flat, Y_pred_tslib
        gc.collect()

        # preview 使用物理尺度数据
        preview_hist_list = X_test_inv_phys[:MAX_PREVIEW].reshape(MAX_PREVIEW, seq_len * n_feat).tolist()
        preview_pred = Y_pred_inv[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).tolist()
        preview_true = Y_test_inv[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).tolist()

        del Y_pred_inv, Y_test_inv
        gc.collect()

        # ── 补丁一（延续）：修复检索元数据反归一化广播错误 ─────────────────
        # 关键问题：topk_hist 被展开为 (n_meta * k_meta)，而 revin_stats 是 (n_test, ...)
        # 解决方案：截取前 n_meta 个统计量，在 axis=0 上重复 k_meta 次对齐
        meta_file = os.path.join(exp_dir, 'retrieval_meta.npz')
        if os.path.exists(meta_file):
            try:
                log.info(f"[{model_name}] Denormalizing retrieval meta to physical scale...")
                meta = np.load(meta_file)
                topk_hist = meta['topk_histories']  # shape: (n_meta, k, seq_len, n_feat)
                topk_futu = meta['topk_futures']    # shape: (n_meta, k, pred_len, n_feat)
                weights = meta['topk_weights']       # shape: (n_meta, k)

                n_meta, k_meta, s_len, f_len = topk_hist.shape
                p_len = topk_futu.shape[2]

                # 关键修复：截取前 n_meta 个统计量，并在 axis=0 上重复 k_meta 次
                sliced_stats = {}
                for key, val in revin_stats_test.items():
                    v_slice = val[:n_meta]  # 取前 n_meta 个测试样本的统计量 (n_meta, 1, ...)
                    sliced_stats[key] = np.repeat(v_slice, k_meta, axis=0)  # 展开对齐

                # 执行反归一化（先 reshape 为 3D）
                hist_3d = topk_hist.reshape(-1, s_len, f_len)  # (n_meta * k, seq_len, n_feat)
                futu_3d = topk_futu.reshape(-1, p_len, f_len)   # (n_meta * k, pred_len, n_feat)

                hist_inv = inverse_revin(hist_3d, sliced_stats, revin_type)
                futu_inv = inverse_revin(futu_3d, sliced_stats, revin_type)

                # 恢复物理量纲（inverse_transform）— 由于 test_set 已删除，使用 Z-score 反归一化
                # 直接用 inverse_revin 的结果（已在 TSLib 空间），不需要再 inverse_transform
                hist_inv_phys = hist_inv.reshape(n_meta, k_meta, s_len, f_len).astype(np.float32)
                futu_inv_phys = futu_inv.reshape(n_meta, k_meta, p_len, f_len).astype(np.float32)

                np.savez(meta_file, topk_histories=hist_inv_phys, topk_futures=futu_inv_phys, topk_weights=weights)
                log.info(f"[{model_name}] Retrieval meta denormalized: hist={hist_inv_phys.shape}, fut={futu_inv_phys.shape}")
            except Exception as e:
                import traceback
                log.error(f"[{model_name}] Meta denorm failed: {e}")
                log.debug(traceback.format_exc())

        # 保存 params.json
        params_out = {
            'model_name': model_name,
            'seq_len': seq_len,
            'pred_len': pred_len,
            'n_features': int(n_feat),
            'revin_type': revin_type,
            **{k: v for k, v in model_params.items() if k != 'device'},
            'device': str(device),
            'dataset': config.get('data_path', 'ETTm1.csv'),
            'features': config.get('features', 'M'),
            'target': config.get('target', 'OT'),
        }
        with open(os.path.join(exp_dir, 'params.json'), 'w', encoding='utf-8') as f:
            json.dump(params_out, f, indent=2, ensure_ascii=False)

        # 保存 metrics.json
        metrics_clean = {}
        for k, v in metrics.items():
            try:
                metrics_clean[k] = float(v) if (v == v) else 0.0
            except Exception:
                metrics_clean[k] = 0.0
        with open(os.path.join(exp_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_clean, f, indent=2)

        # 自动绘图
        try:
            from plotting import plot_comparison_samples
            png_path = os.path.join(exp_dir, 'visualization.png')
            plot_comparison_samples(
                history=np.array(preview_hist_list),
                preds=np.array(preview_pred),
                trues=np.array(preview_true),
                seq_len=seq_len,
                pred_len=pred_len,
                n_features=n_feat,
                model_name=model_name,
                params=params_out,
                save_path=png_path,
                n_samples=9,
                figsize=(12, 10)
            )
            log.info(f"[{model_name}] Visualization saved: {png_path}")
        except Exception as plot_err:
            log.warning(f"[{model_name}] Plotting failed (non-critical): {plot_err}")

        log.info(f"[{model_name}] Done elapsed={elapsed:.1f}s")

        # 追加到 summary_metrics.csv
        _append_to_summary_csv(run_dir, exp_id, model_name, seq_len, pred_len, revin_type,
                               metrics_clean, elapsed, exp_dir)

        return {
            'exp_id': exp_id,
            'exp_dir': exp_dir,
            'config': config,
            'metrics': metrics,
            'status': 'success',
            'elapsed': round(elapsed, 2),
        }

    except Exception as e:
        import traceback
        log.error(f"[{model_name}] Failed: {e}")
        log.debug(traceback.format_exc())

        # 失败时也追加记录
        _append_to_summary_csv(run_dir, exp_id, model_name, seq_len, pred_len, revin_type,
                               {}, 0, exp_dir, status='failed', error=str(e))

        return {
            'exp_id': exp_id,
            'exp_dir': exp_dir,
            'config': config,
            'metrics': {},
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed': round(time.time() - start_time, 2)
        }


def _append_to_summary_csv(
    run_dir: str,
    exp_id: str,
    model_name: str,
    seq_len: int,
    pred_len: int,
    revin_type: str,
    metrics: Dict[str, float],
    elapsed: float,
    exp_dir: str,
    status: str = 'success',
    error: str = ''
):
    """将实验结果追加到 summary_metrics.csv"""
    import pandas as pd

    csv_path = os.path.join(run_dir, 'summary_metrics.csv')
    new_row = {
        'exp_id': exp_id,
        'model': model_name,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'revin_type': revin_type,
        'MAE': metrics.get('MAE', ''),
        'MSE': metrics.get('MSE', ''),
        'RMSE': metrics.get('RMSE', ''),
        'MAPE': metrics.get('MAPE', ''),
        'RSE': metrics.get('RSE', ''),
        'CORR': metrics.get('CORR', ''),
        'status': status,
        'elapsed': elapsed,
        'exp_dir': exp_dir,
        'error': error,
    }

    # 检查文件是否存在
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            # 检查是否已存在该 exp_id（防止重复追加）
            if exp_id in df_existing['exp_id'].values:
                # 更新现有行
                idx = df_existing[df_existing['exp_id'] == exp_id].index[0]
                for col, val in new_row.items():
                    df_existing.loc[idx, col] = val
                df_existing.to_csv(csv_path, index=False)
            else:
                df_new = pd.DataFrame([new_row])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to update summary CSV: {e}")
            # 重新创建
            df_new = pd.DataFrame([new_row])
            df_new.to_csv(csv_path, index=False)
    else:
        df_new = pd.DataFrame([new_row])
        df_new.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV created: {csv_path}")


def _get_default(param: str) -> Any:
    """获取参数默认值（v4.0: V100 32G 极限释放）"""
    defaults = {
        # KNN 基线
        'top_k': 5,
        'weighted': True,
        'predict_chunk_size': 4096,
        # LSH
        'n_hash_funcs': 16,
        'n_tables': 4,
        'hamming_radius': 2,
        'candidate_cap_per_table': 256,
        'candidate_cap_total': 1024,
        # SAX
        'word_size': 8,
        'alphabet_size': 8,
        'epsilon_threshold': 1.0,
        'bucket_top_k': 8,
        # DTWSearch
        'dtw_radius': 5,
        # MatrixProfileSearch
        'subsequence_length': None,
        'normalize': True,
        'train_chunk_size': 4096,
        # TS2VecSearch
        'hidden_dim': 64,
        'epochs': 100,
        'batch_size': 1024,
        'lr': 1e-3,
        'temperature': 0.1,
        # RAGSearch
        'd_model': 256,
        'n_heads': 8,
        'weight_decay': 1e-4,
    }
    return defaults.get(param)


def _make_exp_id(model_name: str, seq_len: int, pred_len: int, config: Dict) -> str:
    """生成实验ID（含模型专属超参数后缀 + revin_type）"""
    dataset = config.get('data_path', 'ETTm1.csv').replace('.csv', '')
    revin = config.get('revin_type', 'none')
    revin_suffix = '_R' + revin[0].upper() if revin and revin != 'none' else ''

    base = dataset + "_" + model_name + "_seq" + str(seq_len) + "_pred" + str(pred_len)

    if model_name == 'PatternSearch':
        return base + "_k" + str(config.get('top_k', 5)) + revin_suffix
    elif model_name == 'LSHSearch':
        return (base
                + "_h" + str(config.get('n_hash_funcs', 16))
                + "_t" + str(config.get('n_tables', 4))
                + revin_suffix)
    elif model_name == 'SAXSearch':
        return (base
                + "_w" + str(config.get('word_size', 8))
                + "_a" + str(config.get('alphabet_size', 8))
                + revin_suffix)
    elif model_name == 'DTWSearch':
        return (base
                + "_k" + str(config.get('top_k', 5))
                + "_r" + str(config.get('dtw_radius', 5))
                + revin_suffix)
    elif model_name == 'MatrixProfileSearch':
        return (base
                + "_k" + str(config.get('top_k', 5))
                + revin_suffix)
    elif model_name == 'TS2VecSearch':
        return (base
                + "_hd" + str(config.get('hidden_dim', 64))
                + "_e" + str(config.get('epochs', 100))
                + "_k" + str(config.get('top_k', 5))
                + revin_suffix)
    elif model_name == 'RAGSearch':
        return (base
                + "_dm" + str(config.get('d_model', 256))
                + "_nh" + str(config.get('n_heads', 8))
                + "_e" + str(config.get('epochs', 100))
                + revin_suffix)
    else:
        return base + revin_suffix


# ============================================================
# 命令行入口（v4.0: 简化为纯工作节点）
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Time-Series Forecasting Baseline (v4.0 - Pure Worker Node)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单次实验（Shell 调度）
  python run.py --model PatternSearch --seq_len 96 --pred_len 96 --run_dir ./results/run_001

  # 带模型专属参数
  python run.py --model RAGSearch --seq_len 96 --pred_len 96 --rag_d_model 256 --rag_epochs 50 --run_dir ./results/run_001
        """
    )

    # 模型选择
    parser.add_argument('--model', type=str, default='PatternSearch',
                       help='Model: PatternSearch, LSHSearch, SAXSearch, DTWSearch, '
                            'MatrixProfileSearch, TS2VecSearch, RAGSearch')

    # 数据参数
    parser.add_argument('--root_path', type=str, default='./ETT_data')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S'])
    parser.add_argument('--target', type=str, default='OT')

    # 序列参数（标量值，不再接收列表）
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction horizon length')

    # PatternSearch
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--weighted', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--predict_chunk_size', type=int, default=4096)

    # LSHSearch
    parser.add_argument('--n_hash_funcs', type=int, default=16)
    parser.add_argument('--n_tables', type=int, default=4)
    parser.add_argument('--hamming_radius', type=int, default=2)
    parser.add_argument('--candidate_cap_per_table', type=int, default=256)
    parser.add_argument('--candidate_cap_total', type=int, default=1024)
    parser.add_argument('--lsh_weighted', type=lambda x: x.lower() == 'true', default=False)

    # SAXSearch
    parser.add_argument('--word_size', type=int, default=8)
    parser.add_argument('--alphabet_size', type=int, default=8)
    parser.add_argument('--epsilon_threshold', type=float, default=1.0)
    parser.add_argument('--bucket_top_k', type=int, default=8)
    parser.add_argument('--sax_weighted', type=lambda x: x.lower() == 'true', default=True)

    # DTWSearch
    parser.add_argument('--dtw_radius', type=int, default=5)

    # MatrixProfileSearch
    parser.add_argument('--subsequence_length', type=int, default=None)
    parser.add_argument('--mp_normalize', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--mp_chunk_size', type=int, default=4096)
    parser.add_argument('--mp_train_chunk_size', type=int, default=2048)

    # TS2VecSearch
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--ts2vec_epochs', type=int, default=100)
    parser.add_argument('--ts2vec_batch_size', type=int, default=1024)
    parser.add_argument('--ts2vec_lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.1)

    # RAGSearch
    parser.add_argument('--rag_d_model', type=int, default=256)
    parser.add_argument('--rag_n_heads', type=int, default=8)
    parser.add_argument('--rag_epochs', type=int, default=100)
    parser.add_argument('--rag_batch_size', type=int, default=1024)
    parser.add_argument('--rag_lr', type=float, default=1e-3)
    parser.add_argument('--rag_weight_decay', type=float, default=1e-4)

    # 执行参数
    parser.add_argument('--run_dir', type=str, required=True,
                       help='Run output directory (managed by shell script)')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--revin_type', type=str, default='none',
                       choices=['none', 'temporal', 'feature', 'dual'])

    return parser.parse_args()


def main():
    args = parse_args()

    # 确保 run_dir 存在
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, 'logs'), exist_ok=True)

    # 构建配置
    config = {
        'model_name': args.model,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'revin_type': args.revin_type,
        'root_path': args.root_path,
        'data_path': args.data_path,
        'features': args.features,
        'target': args.target,
        'use_gpu': args.use_gpu,
        'run_dir': os.path.abspath(args.run_dir),
    }

    # 添加模型专属参数
    model_params = MODEL_REGISTRY.get(args.model, {}).get('params', [])
    for param in model_params:
        arg_name = param
        # 处理 RAGSearch 特有前缀
        if args.model == 'RAGSearch' and param in ['d_model', 'n_heads', 'epochs', 'batch_size', 'lr', 'weight_decay']:
            arg_name = f'rag_{param}'
        # 处理 TS2VecSearch 特有前缀
        elif args.model == 'TS2VecSearch' and param in ['epochs', 'batch_size', 'lr']:
            arg_name = f'ts2vec_{param}'
        # 处理 MatrixProfileSearch 特有参数
        elif args.model == 'MatrixProfileSearch':
            if param == 'subsequence_length':
                arg_name = 'subsequence_length'
            elif param == 'normalize':
                arg_name = 'mp_normalize'
            elif param == 'predict_chunk_size':
                arg_name = 'mp_chunk_size'
            elif param == 'train_chunk_size':
                arg_name = 'mp_train_chunk_size'
        # 处理通用参数
        elif param == 'weighted':
            if args.model == 'LSHSearch':
                arg_name = 'lsh_weighted'
            elif args.model == 'SAXSearch':
                arg_name = 'sax_weighted'
        elif param == 'predict_chunk_size':
            if args.model == 'DTWSearch':
                arg_name = 'predict_chunk_size'

        if hasattr(args, arg_name):
            val = getattr(args, arg_name)
            if val is not None:
                config[param] = val

    # 打印启动信息
    logger.info("=" * 60)
    logger.info(f"Time-Series Forecasting Worker Node [v4.0]")
    logger.info(f"Model: {args.model}")
    logger.info(f"Seq: {args.seq_len}, Pred: {args.pred_len}")
    logger.info(f"RevIN: {args.revin_type}")
    logger.info(f"Run Dir: {args.run_dir}")
    logger.info("=" * 60)

    # 执行实验
    result = run_single_experiment(config)

    # 打印结果摘要
    if result['status'] == 'success':
        m = result['metrics']
        logger.info("=" * 60)
        logger.info(f"Result: SUCCESS")
        logger.info(f"MAE:  {m.get('MAE', 0):.4f}")
        logger.info(f"MSE:  {m.get('MSE', 0):.4f}")
        logger.info(f"RMSE: {m.get('RMSE', 0):.4f}")
        logger.info(f"Time:  {result['elapsed']:.1f}s")
        logger.info("=" * 60)
    else:
        logger.error(f"Result: FAILED - {result.get('error', 'unknown')}")

    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
