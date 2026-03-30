"""
统一入口脚本 - 时序预测基线模型系统

v3.4 核心变更：
1. Run-Level 时间戳目录隔离：每次批量运行生成独立 run_{timestamp}/
2. Dual-Dimension RevIN：revin_type 支持 none / temporal / feature / dual
3. 目录规范：每个实验独立文件夹 run_{timestamp}/{exp_id}/
4. Summary CSV：ExperimentRunner 结束时汇总所有实验指标到 summary_metrics.csv
5. V100 32G 超参数大释放

Usage:
    python run.py --model all --seq_len 96 --pred_len 48 --revin_type dual
    python run.py --model PatternSearch --revin_type temporal --top_k 5
"""

import os
import sys
import gc
import json
import time
import logging
import psutil
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# 全局配置（Run-Level 时间戳隔离）
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 默认 RESULTS_DIR = results/（单次快速运行）
# ExperimentRunner 初始化时会升格为 run_{timestamp}/
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 日志配置（路径由 setup_run_dir 动态注入）
# ============================================================

def _setup_logger(exp_id: Optional[str] = None) -> logging.Logger:
    """为每个实验配置独立日志器（写入 {RUN_DIR}/logs/{exp_id}.log）"""
    log_dir = os.path.join(RUN_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log = logging.getLogger(__name__ + ('_' + exp_id if exp_id else ''))
    log.setLevel(logging.INFO)
    log.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    if exp_id:
        fh = logging.FileHandler(os.path.join(log_dir, exp_id + '.log'), encoding='utf-8')
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

# ─────────────────────────────────────────────────────────────
# Run-Level 全局目录（setup_run_dir 初始化后生效）
# ─────────────────────────────────────────────────────────────
RUN_DIR = RESULTS_DIR  # fallback


def setup_run_dir() -> str:
    """
    生成带时间戳的运行目录，并将全局 RUN_DIR 指向它。
    所有 experiment_log.json / summary_metrics.csv / {exp_id}/ 均写入此目录。
    """
    global RUN_DIR
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_DIR = os.path.join(PROJECT_ROOT, 'results', f'run_{ts}')
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, 'logs'), exist_ok=True)
    return RUN_DIR


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
    """数值安全：将 std < threshold 的位置强制置 1.0，防止除零放大"""
    return np.where(std < threshold, 1.0, std)


def apply_revin(
    X: np.ndarray,
    revin_type: str,
    prefix: str = 'X'
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Dual-Dimension RevIN 归一化

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
        mean = np.mean(X, axis=-1, keepdims=True)
        std = _safe_std(np.std(X, axis=-1, keepdims=True))
        X_out = ((X - mean) / std).astype(np.float32)
        stats = {'mean_f': mean, 'std_f': std}
        logger.info(f"[RevIN-{prefix}] feature: mean={mean.shape}, std={std.shape}")

    elif revin_type == 'dual':
        mean_f = np.mean(X, axis=-1, keepdims=True)
        std_f = _safe_std(np.std(X, axis=-1, keepdims=True))
        X_f = ((X - mean_f) / std_f).astype(np.float32)
        mean_t = np.mean(X_f, axis=1, keepdims=True)
        std_t = _safe_std(np.std(X_f, axis=1, keepdims=True))
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
    """Dual-Dimension RevIN 反归一化"""
    if revin_type == 'none':
        return Y_norm.astype(np.float32)

    elif revin_type == 'temporal':
        mean_t = revin_stats['mean_t']
        std_t = revin_stats['std_t']
        return (Y_norm * std_t + mean_t).astype(np.float32)

    elif revin_type == 'feature':
        mean_f = revin_stats['mean_f']
        std_f = revin_stats['std_f']
        return (Y_norm * std_f + mean_f).astype(np.float32)

    elif revin_type == 'dual':
        mean_t = revin_stats['mean_t']
        std_t = revin_stats['std_t']
        mean_f = revin_stats['mean_f']
        std_f = revin_stats['std_f']
        Y_t = Y_norm * std_t + mean_t
        return (Y_t * std_f + mean_f).astype(np.float32)

    return Y_norm.astype(np.float32)


# ============================================================
# 核心计算逻辑（可独立复用）
# ============================================================

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单次实验

    v3.4 变更：
    1. 实验结果写入 RUN_DIR/{exp_id}/（由 setup_run_dir 注入）
    2. Dual-Dimension RevIN
    3. 指标在归一化空间计算
    4. 实验结束时自动调用绘图函数生成 PNG
    """
    import numpy as np
    from data_provider.data_loader import get_data, get_X_Y_from_dataset
    from utils.metrics import calculate_all_metrics

    start_time = time.time()
    model_name = config['model_name']
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    revin_type = config.get('revin_type', 'none')

    # 生成 exp_id（写入 RUN_DIR）
    exp_id = _make_exp_id(model_name, seq_len, pred_len, config)
    exp_dir = os.path.join(RUN_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # 实验级日志器
    log = _setup_logger(exp_id)
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
        # GPU 自动检测
        if torch.cuda.is_available():
            device = 'cuda'
            log.info(f"[{model_name}] CUDA detected, device=cuda")
        else:
            device = 'cpu'
            if config.get('use_gpu', False):
                log.warning(f"[{model_name}] GPU requested but unavailable, fallback to CPU")
        model_params['device'] = device
        log.info(f"[{model_name}] device={device}, revin_type={revin_type}")

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

        # TSLib Y 截断
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
        Y_pred_norm = model.predict(X_test_norm)

        del model, X_test_norm
        gc.collect()

        # 指标计算（在归一化空间）
        metrics = calculate_all_metrics(Y_pred_norm, Y_test_3d)

        elapsed = time.time() - start_time
        log.info(f"[{model_name}] Norm-space MAE={metrics.get('MAE', 0):.4f} "
                 f"MSE={metrics.get('MSE', 0):.4f} elapsed={elapsed:.1f}s")

        # 反归一化
        Y_pred = inverse_revin(Y_pred_norm, revin_stats_test, revin_type)
        del Y_pred_norm
        gc.collect()

        # inverse_transform + 保存 .npy
        n_test, p_len, _ = Y_pred.shape
        Y_pred_flat = Y_pred.reshape(-1, n_feat)
        Y_test_flat = Y_test_3d.reshape(-1, n_feat)

        Y_pred_inv = test_set.inverse_transform(Y_pred_flat)
        Y_test_inv = test_set.inverse_transform(Y_test_flat)

        Y_pred_inv = Y_pred_inv.reshape(n_test, p_len, n_feat).astype(np.float32)
        Y_test_inv = Y_test_inv.reshape(n_test, p_len, n_feat).astype(np.float32)

        np.save(os.path.join(exp_dir, 'preds.npy'), Y_pred_inv)
        np.save(os.path.join(exp_dir, 'trues.npy'), Y_test_inv)
        np.save(os.path.join(exp_dir, 'X_test.npy'), history_preview_raw.reshape(MAX_PREVIEW, -1))

        del test_set, Y_pred_flat, Y_test_flat
        gc.collect()

        # JSON preview
        preview_hist_list = history_preview_raw.reshape(
            MAX_PREVIEW, seq_len * n_feat
        ).tolist()
        preview_pred = Y_pred_inv[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).tolist()
        preview_true = Y_test_inv[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).tolist()

        del Y_pred, Y_pred_inv, Y_test_inv
        gc.collect()

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
        return {
            'exp_id': exp_id,
            'exp_dir': exp_dir,
            'config': config,
            'metrics': metrics,
            'status': 'success',
            'elapsed': round(elapsed, 2),
            'preview': {
                'preds': preview_pred,
                'trues': preview_true,
                'history': preview_hist_list,
                'count': min(len(preview_pred), len(preview_true)),
            },
            'npy_file': {
                'preds': os.path.join(exp_dir, 'preds.npy'),
                'trues': os.path.join(exp_dir, 'trues.npy'),
                'x_test': os.path.join(exp_dir, 'X_test.npy'),
            }
        }

    except Exception as e:
        import traceback
        log.error(f"[{model_name}] Failed: {e}")
        log.debug(traceback.format_exc())
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


def _get_default(param: str) -> Any:
    """获取参数默认值（v3.4: V100 32G 大释放）"""
    defaults = {
        # ── KNN 基线 ─────────────────────────────────────────
        'top_k': 5,
        'weighted': True,
        'predict_chunk_size': 2048,
        # ── LSH ───────────────────────────────────────────────
        'n_hash_funcs': 16,
        'n_tables': 4,
        'hamming_radius': 2,
        'candidate_cap_per_table': 256,
        'candidate_cap_total': 1024,
        # ── SAX ───────────────────────────────────────────────
        'word_size': 8,
        'alphabet_size': 8,
        'epsilon_threshold': 1.0,
        'bucket_top_k': 8,
        # ── DTWSearch（v3.4: chunk 512 → 1024）────────────────
        'dtw_radius': 5,
        # ── MatrixProfileSearch（v3.4: chunk 512 → 4096）───────
        'subsequence_length': None,
        'normalize': True,
        'train_chunk_size': 2048,
        # ── TS2VecSearch（v3.4: epochs 10→50, batch 128→256）─
        'hidden_dim': 64,
        'epochs': 50,
        'batch_size': 256,
        'lr': 1e-3,
        'temperature': 0.1,
        # ── RAGSearch（v3.4: d_model 32→128, heads 4→8, e 10→50, bs 128→256）
        'd_model': 128,
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
                + "_e" + str(config.get('epochs', 50))
                + "_k" + str(config.get('top_k', 5))
                + revin_suffix)
    elif model_name == 'RAGSearch':
        return (base
                + "_dm" + str(config.get('d_model', 128))
                + "_nh" + str(config.get('n_heads', 8))
                + "_e" + str(config.get('epochs', 50))
                + revin_suffix)
    else:
        return base + revin_suffix


def _expand_configs(model_list: List[str], seq_lens: List[int], pred_lens: List[int],
                    all_configs: List[Dict]) -> List[Dict[str, Any]]:
    """展开参数网格"""
    configs = []
    for model in model_list:
        for seq in seq_lens:
            for pred in pred_lens:
                for cfg in all_configs:
                    if cfg['model_name'] == model:
                        config = cfg.copy()
                        config['seq_len'] = seq
                        config['pred_len'] = pred
                        configs.append(config)
    return configs


# ============================================================
# 实验运行器
# ============================================================

class ExperimentRunner:
    """
    实验运行器（可导入复用）

    v3.4 变更：
    - __init__ 中调用 setup_run_dir() 生成带时间戳的 RUN_DIR
    - 所有日志 / 实验文件夹 / summary 全部写入 RUN_DIR
    """

    MEMORY_THRESHOLD = 0.85

    def __init__(self, args):
        self.args = args
        self.results = []
        self.run_dir = setup_run_dir()          # ← 时间戳隔离
        global RUN_DIR
        RUN_DIR = self.run_dir                  # ← 注入全局
        self._memory_check()

    def _memory_check(self) -> bool:
        try:
            mem = psutil.virtual_memory()
            usage = mem.percent / 100.0
            avail_gb = mem.available / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            logger.info(f"[Memory] {usage:.1%} used ({total_gb:.1f}G total, {avail_gb:.1f}G available)")

            if usage >= self.MEMORY_THRESHOLD:
                logger.warning(f"[Memory] {usage:.1%} >= {self.MEMORY_THRESHOLD:.1%}, "
                               f"parallel mode degraded to sequential")
                return False
            return True
        except Exception as e:
            logger.warning(f"[Memory] check failed ({e}), conservative sequential mode")
            return False

    def build_configs(self) -> List[Dict[str, Any]]:
        models = []
        if self.args.model == 'all':
            models = list(MODEL_REGISTRY.keys())
        else:
            models = [m.strip() for m in self.args.model.split(',')]

        seq_lens = self.args.seq_len if self.args.seq_len else [96]
        pred_lens = self.args.pred_len if self.args.pred_len else [48]

        all_configs = []

        for m in models:
            cfg = {'model_name': m}

            # PatternSearch
            if m == 'PatternSearch':
                cfg.update({
                    'top_k': self.args.top_k,
                    'weighted': self.args.weighted,
                    'predict_chunk_size': getattr(self.args, 'predict_chunk_size', 2048),
                })

            # LSHSearch
            elif m == 'LSHSearch':
                cfg.update({
                    'n_hash_funcs': self.args.n_hash_funcs,
                    'n_tables': self.args.n_tables,
                    'hamming_radius': self.args.hamming_radius,
                    'candidate_cap_per_table': self.args.candidate_cap_per_table,
                    'candidate_cap_total': self.args.candidate_cap_total,
                    'weighted': self.args.lsh_weighted,
                })

            # SAXSearch
            elif m == 'SAXSearch':
                cfg.update({
                    'word_size': self.args.word_size,
                    'alphabet_size': self.args.alphabet_size,
                    'epsilon_threshold': self.args.epsilon_threshold,
                    'bucket_top_k': self.args.bucket_top_k,
                    'weighted': self.args.sax_weighted,
                })

            # DTWSearch（v3.4: chunk 512 → 1024）
            elif m == 'DTWSearch':
                cfg.update({
                    'top_k': self.args.top_k,
                    'dtw_radius': getattr(self.args, 'dtw_radius', 5),
                    'weighted': self.args.weighted,
                    'predict_chunk_size': getattr(self.args, 'predict_chunk_size', 1024),
                })

            # MatrixProfileSearch（v3.4: chunk 512 → 4096）
            elif m == 'MatrixProfileSearch':
                cfg.update({
                    'top_k': self.args.top_k,
                    'subsequence_length': getattr(self.args, 'subsequence_length', None),
                    'normalize': getattr(self.args, 'mp_normalize', True),
                    'predict_chunk_size': getattr(self.args, 'mp_chunk_size', 4096),
                    'train_chunk_size': getattr(self.args, 'mp_train_chunk_size', 2048),
                })

            # TS2VecSearch（v3.4: epochs 10→50, batch 128→256）
            elif m == 'TS2VecSearch':
                cfg.update({
                    'hidden_dim': getattr(self.args, 'hidden_dim', 64),
                    'epochs': getattr(self.args, 'ts2vec_epochs', 50),
                    'batch_size': getattr(self.args, 'ts2vec_batch_size', 256),
                    'top_k': self.args.top_k,
                    'lr': getattr(self.args, 'ts2vec_lr', 1e-3),
                    'temperature': getattr(self.args, 'temperature', 0.1),
                })

            # RAGSearch（v3.4: d_model 32→128, heads 4→8, e 10→50, bs 128→256）
            elif m == 'RAGSearch':
                cfg.update({
                    'd_model': getattr(self.args, 'rag_d_model', 128),
                    'n_heads': getattr(self.args, 'rag_n_heads', 8),
                    'epochs': getattr(self.args, 'rag_epochs', 50),
                    'batch_size': getattr(self.args, 'rag_batch_size', 256),
                    'lr': getattr(self.args, 'rag_lr', 1e-3),
                    'weight_decay': getattr(self.args, 'rag_weight_decay', 1e-4),
                })

            all_configs.append(cfg)

        configs = _expand_configs(models, seq_lens, pred_lens, all_configs)

        for cfg in configs:
            cfg['root_path'] = self.args.root_path
            cfg['data_path'] = self.args.data_path
            cfg['features'] = self.args.features
            cfg['target'] = self.args.target
            cfg['use_gpu'] = getattr(self.args, 'use_gpu', False)
            cfg['revin_type'] = getattr(self.args, 'revin_type', 'none')

        return configs

    def run(self):
        """运行实验（串行 / 并行自适应）"""
        configs = self.build_configs()
        total = len(configs)

        logger.info("=" * 60)
        logger.info("Time-Series Forecasting Baseline System  [v3.4]")
        logger.info(f"Run directory: {self.run_dir}")
        logger.info("=" * 60)
        unique_models = len(set(c['model_name'] for c in configs))
        logger.info(f"Models: {configs[0]['model_name'] if unique_models == 1 else 'all'} ({total} experiments)")
        logger.info("=" * 60)

        use_parallel = self.args.parallel and total > 1
        memory_safe = self._memory_check()

        if use_parallel and memory_safe:
            self._run_parallel(configs, total)
        else:
            if use_parallel and not memory_safe:
                logger.warning("Parallel request blocked by memory protection, sequential mode")
            self._run_sequential(configs, total)

        self._save_log()
        return self.results

    def _run_sequential(self, configs: List[Dict], total: int):
        """串行执行，单个失败不中断"""
        logger.info(f"Sequential execution of {total} experiments...\n")

        for i, cfg in enumerate(configs):
            model_name = cfg['model_name']
            logger.info(f"[{i+1}/{total}] {model_name} seq={cfg['seq_len']} pred={cfg['pred_len']}")

            try:
                result = run_single_experiment(cfg)
            except Exception as e:
                logger.error(f"[{i+1}/{total}] {model_name} uncaught exception: {e}")
                result = {
                    'config': cfg,
                    'metrics': {},
                    'status': 'failed',
                    'error': f'uncaught exception: {e}',
                    'elapsed': 0
                }

            self.results.append(result)
            gc.collect()

            if result['status'] == 'success':
                m = result['metrics']
                logger.info(f"  -> OK MAE={m.get('MAE', 0):.4f} "
                            f"MSE={m.get('MSE', 0):.4f} elapsed={result['elapsed']:.1f}s")
            else:
                logger.warning(f"  -> FAIL: {result.get('error', 'unknown')}")

    def _run_parallel(self, configs: List[Dict], total: int):
        """并行执行，单个失败不中断"""
        n_workers = min(self.args.n_workers, total, 4)
        logger.info(f"Parallel execution: {n_workers} workers\n")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_experiment, cfg): i
                        for i, cfg in enumerate(configs)}

            for future in as_completed(futures):
                idx = futures[future]
                cfg = configs[idx]
                logger.info(f"[{idx+1}/{total}] {cfg['model_name']} seq={cfg['seq_len']} pred={cfg['pred_len']}")

                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"[{idx+1}/{total}] {cfg['model_name']} future exception: {e}")
                    result = {
                        'config': cfg,
                        'metrics': {},
                        'status': 'failed',
                        'error': f'future exception: {e}',
                        'elapsed': 0
                    }

                self.results.append(result)
                gc.collect()

                if result['status'] == 'success':
                    m = result['metrics']
                    logger.info(f"  -> OK MAE={m.get('MAE', 0):.4f} "
                                f"MSE={m.get('MSE', 0):.4f} elapsed={result['elapsed']:.1f}s")
                else:
                    logger.warning(f"  -> FAIL: {result.get('error', 'unknown')}")

    def _save_log(self):
        """保存实验日志 + summary_metrics.csv 到 RUN_DIR"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'run_dir': self.run_dir,
            'metadata': {
                'dataset': self.args.data_path,
                'features': self.args.features,
                'total': len(self.results),
            },
            'experiments': []
        }

        csv_rows = []

        for r in self.results:
            cfg = r['config']
            m = r['metrics']

            exp_entry = {
                'config': cfg,
                'metrics': m,
                'status': r['status'],
                'elapsed': r.get('elapsed', 0),
            }

            if r['status'] == 'success' and 'preview' in r:
                exp_entry['preview'] = r['preview']
                exp_entry['npy_file'] = r.get('npy_file', {})

            if r['status'] == 'failed':
                exp_entry['error'] = r.get('error', '')

            log['experiments'].append(exp_entry)

            csv_rows.append({
                'exp_id': r.get('exp_id', ''),
                'model': cfg.get('model_name', ''),
                'seq_len': cfg.get('seq_len', 0),
                'pred_len': cfg.get('pred_len', 0),
                'revin_type': cfg.get('revin_type', 'none'),
                'MAE': m.get('MAE', ''),
                'MSE': m.get('MSE', ''),
                'RMSE': m.get('RMSE', ''),
                'MAPE': m.get('MAPE', ''),
                'RSE': m.get('RSE', ''),
                'CORR': m.get('CORR', ''),
                'status': r['status'],
                'elapsed': r.get('elapsed', ''),
                'exp_dir': r.get('exp_dir', ''),
            })

        # 写入 RUN_DIR（时间戳隔离目录）
        log_path = os.path.join(self.run_dir, 'experiment_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        import pandas as pd
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_path = os.path.join(self.run_dir, 'summary_metrics.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info("Summary CSV saved: " + csv_path)

            # 汇总柱状图（英文，无乱码）
            try:
                from plotting import plot_summary_bar
                bar_path = os.path.join(self.run_dir, 'summary_MAE_bar.png')
                plot_summary_bar(csv_path, metric='MAE', save_path=bar_path)
                logger.info("Summary bar chart saved: " + bar_path)
            except Exception:
                pass

        self._print_summary()

    def _print_summary(self):
        """打印结果摘要"""
        success = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']

        logger.info("=" * 70)
        logger.info("Time-Series Forecasting Baseline  [v3.4]")
        logger.info(f"Run directory: {self.run_dir}")
        logger.info("=" * 70)
        logger.info("exp_id                                    | model            | revin | seq  | pred |  MAE   |  MSE")
        logger.info("-" * 70)
        for r in self.results:
            cfg = r['config']
            m = r['metrics']
            eid = (r.get('exp_id', '') or '')[:40]
            revin = cfg.get('revin_type', 'none')[:6]
            mae = "%.4f" % m.get('MAE', 0)
            mse = "%.4f" % m.get('MSE', 0)
            sep = "OK" if r['status'] == 'success' else "FAIL"
            logger.info("%-42s %-16s %-7s %-5d %-5d %-8s %s (%s)"
                        % (eid, cfg.get('model_name', ''), revin,
                           cfg.get('seq_len', 0), cfg.get('pred_len', 0),
                           mae, mse, sep))

        logger.info("-" * 70)
        logger.info(f"Done: {len(success)}/{len(self.results)} OK, {len(failed)} FAIL")
        logger.info(f"Log: {os.path.join(self.run_dir, 'experiment_log.json')}")
        logger.info(f"CSV: {os.path.join(self.run_dir, 'summary_metrics.csv')}")


# ============================================================
# 仪表盘启动
# ============================================================

def launch_dashboard():
    """启动 Streamlit 仪表盘"""
    import subprocess
    import webbrowser

    dashboard_path = os.path.join(PROJECT_ROOT, 'dashboard', 'app.py')

    if not os.path.exists(dashboard_path):
        logger.error(f"Dashboard not found: {dashboard_path}")
        return

    cmd = [
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.port', '8501', '--server.headless', 'true',
        '--server.address', '0.0.0.0'
    ]
    try:
        subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        logger.info("Dashboard launched: http://localhost:8501")
        time.sleep(2)
        webbrowser.open('http://localhost:8501')
    except Exception as e:
        logger.error(f"Dashboard launch failed: {e}")
        logger.info(f"Please run manually: {' '.join(cmd)}")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Time-Series Forecasting Baseline')

    # 模式选择
    parser.add_argument('--model', type=str, default='PatternSearch',
                       help='Model: PatternSearch, LSHSearch, SAXSearch, DTWSearch, '
                            'MatrixProfileSearch, TS2VecSearch, RAGSearch, all')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch Streamlit dashboard after experiments')
    parser.add_argument('--skip_run', action='store_true',
                       help='Skip experiments, only launch dashboard')

    # 数据参数
    parser.add_argument('--root_path', type=str, default='./ETT_data')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S'])
    parser.add_argument('--target', type=str, default='OT')

    # 序列参数
    parser.add_argument('--seq_len', type=int, nargs='+', default=[96])
    parser.add_argument('--pred_len', type=int, nargs='+', default=[48])

    # PatternSearch
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--weighted', type=lambda x: x.lower() == 'true', default=True)

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

    # ── v3.0+ 模型参数 ─────────────────────────────────────────

    # 通用参数
    parser.add_argument('--predict_chunk_size', type=int, default=2048,
                       help='Prediction chunk size (DTWSearch default: 1024)')

    # DTWSearch
    parser.add_argument('--dtw_radius', type=int, default=5,
                       help='DTW Sakoe-Chiba constraint radius, default 5')

    # MatrixProfileSearch（v3.4: chunk 512 → 4096）
    parser.add_argument('--subsequence_length', type=int, default=None,
                       help='MatrixProfile subsequence length (None=seq_len)')
    parser.add_argument('--mp_normalize', type=lambda x: x.lower() == 'true', default=True,
                       help='MatrixProfile Z-normalization, default True')
    parser.add_argument('--mp_chunk_size', type=int, default=4096,
                       help='MatrixProfile predict chunk size (V100 32G), default 4096')
    parser.add_argument('--mp_train_chunk_size', type=int, default=2048,
                       help='MatrixProfile train chunk size, default 2048')

    # TS2VecSearch（v3.4: epochs 10→50, batch 128→256）
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='TS2Vec / RAG hidden dimension, default 64')
    parser.add_argument('--ts2vec_epochs', type=int, default=50,
                       help='TS2Vec training epochs, default 50')
    parser.add_argument('--ts2vec_batch_size', type=int, default=256,
                       help='TS2Vec batch size, default 256')
    parser.add_argument('--ts2vec_lr', type=float, default=1e-3,
                       help='TS2Vec learning rate, default 1e-3')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Contrastive loss temperature, default 0.1')

    # RAGSearch（v3.4: d_model 32→128, heads 4→8, e 10→50, bs 128→256）
    parser.add_argument('--rag_d_model', type=int, default=128,
                       help='RAG Cross-Attention hidden dim, default 128')
    parser.add_argument('--rag_n_heads', type=int, default=8,
                       help='RAG attention heads, default 8')
    parser.add_argument('--rag_epochs', type=int, default=50,
                       help='RAG training epochs, default 50')
    parser.add_argument('--rag_batch_size', type=int, default=256,
                       help='RAG batch size, default 256')
    parser.add_argument('--rag_lr', type=float, default=1e-3,
                       help='RAG learning rate, default 1e-3')
    parser.add_argument('--rag_weight_decay', type=float, default=1e-4,
                       help='RAG weight decay, default 1e-4')

    # 执行参数
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel execution (auto-degrades if memory > 85%%)')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='Parallel worker count (max 4)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU when torch.cuda is available')
    parser.add_argument('--revin_type', type=str, default='none',
                       choices=['none', 'temporal', 'feature', 'dual'],
                       help='RevIN type: none / temporal / feature / dual')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip_run:
        launch_dashboard()
        sys.exit(0)

    # ExperimentRunner.__init__ 调用 setup_run_dir() 生成时间戳目录
    runner = ExperimentRunner(args)
    runner.run()

    if args.dashboard:
        launch_dashboard()


if __name__ == '__main__':
    main()
