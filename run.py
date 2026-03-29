"""
统一入口脚本 - 时序预测基线模型系统

v3.0 核心变更：
1. Dual-Dimension RevIN：revin_type 支持 none / temporal / feature / dual
   - temporal:  (X - mean(X,axis=1)) / (std(X,axis=1) + eps)   — _instance_ 归一化
   - feature:  (X - mean(X,axis=-1)) / (std(X,axis=-1) + eps) — _channel_ 归一化
   - dual:     先 feature 再 temporal，预测后 inverse_dual 反归一化
   - 数值安全：std < 1e-5 时强制置 1.0
2. 目录规范：每个实验独立文件夹 results/{exp_id}/
3. 自动绘图：实验结束时调用 plot_comparison_samples 生成 PNG
4. Summary CSV：ExperimentRunner 结束时汇总所有实验指标到 summary_metrics.csv

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
# 全局日志配置（带时间戳 + FileHandler 持久化）
# ============================================================

def _setup_logger(exp_id: Optional[str] = None) -> logging.Logger:
    """为每个实验配置独立日志器（写入 results/logs/{exp_id}.log）"""
    log_dir = os.path.join(RESULTS_DIR, 'logs')
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

# ============================================================
# 全局配置
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)

# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY = {
    # ── 传统记忆检索 ──────────────────────────────────────────
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
    # ── v3.0 新增检索模型 ───────────────────────────────────
    'DTWSearch': {
        'class': None,
        'params': ['top_k', 'dtw_radius', 'weighted', 'predict_chunk_size']
    },
    'MatrixProfileSearch': {
        'class': None,
        'params': ['top_k', 'subsequence_length', 'normalize']
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
# 核心计算逻辑（可独立复用）
# ============================================================

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
        # 维度 A：沿 axis=1（时间维度）归一化
        mean = np.mean(X, axis=1, keepdims=True)
        std = _safe_std(np.std(X, axis=1, keepdims=True))
        X_out = ((X - mean) / std).astype(np.float32)
        stats = {'mean_t': mean, 'std_t': std}
        logger.info(f"[RevIN-{prefix}] temporal: mean={mean.shape}, std={std.shape}")

    elif revin_type == 'feature':
        # 维度 B：沿 axis=-1（特征维度）归一化
        mean = np.mean(X, axis=-1, keepdims=True)
        std = _safe_std(np.std(X, axis=-1, keepdims=True))
        X_out = ((X - mean) / std).astype(np.float32)
        stats = {'mean_f': mean, 'std_f': std}
        logger.info(f"[RevIN-{prefix}] feature: mean={mean.shape}, std={std.shape}")

    elif revin_type == 'dual':
        # 维度 B 先：feature-wise（对齐不同特征的量级）
        mean_f = np.mean(X, axis=-1, keepdims=True)
        std_f = _safe_std(np.std(X, axis=-1, keepdims=True))
        X_f = ((X - mean_f) / std_f).astype(np.float32)
        # 维度 A 再：temporal-wise（消除时间趋势）
        mean_t = np.mean(X_f, axis=1, keepdims=True)
        std_t = _safe_std(np.std(X_f, axis=1, keepdims=True))
        X_out = ((X_f - mean_t) / std_t).astype(np.float32)
        stats = {'mean_f': mean_f, 'std_f': std_f, 'mean_t': mean_t, 'std_t': std_t}
        logger.info(f"[RevIN-{prefix}] dual: feature {mean_f.shape} -> temporal {mean_t.shape}")

    else:
        logger.warning(f"[RevIN-{prefix}] 未知 revin_type='{revin_type}'，跳过归一化")

    return X_out, stats


def inverse_revin(
    Y_norm: np.ndarray,
    revin_stats: Dict[str, np.ndarray],
    revin_type: str
) -> np.ndarray:
    """
    Dual-Dimension RevIN 反归一化（仅 RevIN 路径调用）
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
        return (Y_norm * std_f + mean_f).astype(np.float32)

    elif revin_type == 'dual':
        mean_t = revin_stats['mean_t']
        std_t = revin_stats['std_t']
        mean_f = revin_stats['mean_f']
        std_f = revin_stats['std_f']
        # 先反 temporal，再反 feature（与 forward 顺序相反）
        Y_t = Y_norm * std_t + mean_t
        return (Y_t * std_f + mean_f).astype(np.float32)

    return Y_norm.astype(np.float32)


# ============================================================
# 核心计算逻辑（可独立复用）
# ============================================================

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单次实验

    v3.0 核心变更：
    1. Dual-Dimension RevIN：revin_type in {none, temporal, feature, dual}
    2. 目录规范：每个实验结果存入 results/{exp_id}/
    3. 数值安全：std < 1e-5 → 1.0
    4. 指标在归一化空间计算（inverse_transform 之前）
    5. 实验结束时自动调用绘图函数生成 PNG
    """
    import numpy as np
    from data_provider.data_loader import get_data, get_X_Y_from_dataset
    from utils.metrics import calculate_all_metrics

    start_time = time.time()
    model_name = config['model_name']
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    revin_type = config.get('revin_type', 'none')

    # 生成 exp_id（必须先于所有文件操作）
    exp_id = _make_exp_id(model_name, seq_len, pred_len, config)
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # 实验级日志器（写入 results/logs/{exp_id}.log）
    log = _setup_logger(exp_id)
    log.info("=" * 60)
    log.info(f"实验 {exp_id} 启动")
    log.info(f"RevIN 类型: {revin_type}")
    log.info("=" * 60)

    try:
        import_models()
        ModelClass = MODEL_REGISTRY[model_name]['class']

        model_params = {}
        for param in MODEL_REGISTRY[model_name]['params']:
            model_params[param] = config.get(param, _get_default(param))

        import torch
        # ── GPU 自动检测：CUDA 可用时自动升格（不受 --use_gpu 限制）─────────
        if torch.cuda.is_available():
            device = 'cuda'
            log.info(f"[{model_name}] CUDA 可用，device=cuda")
        else:
            device = 'cpu'
            if config.get('use_gpu', False):
                log.warning(f"[{model_name}] 请求 GPU 但不可用，回退 CPU")
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

        log.info(f"[{model_name}] 加载数据 seq={seq_len} pred={pred_len}")
        train_set, val_set, test_set = get_data(args)
        X_train, Y_train = get_X_Y_from_dataset(train_set)
        X_test, Y_test = get_X_Y_from_dataset(test_set)

        log.info(f"[{model_name}] 原始: X_train={X_train.shape}, Y_train={Y_train.shape}, "
                 f"X_test={X_test.shape}, Y_test={Y_test.shape}")

        # ── TSLib Y 截断 ──────────────────────────────────────────
        if Y_train.ndim == 3:
            Y_train = Y_train[:, -pred_len:, :]
        elif Y_train.ndim == 2:
            Y_train = Y_train[:, -pred_len:]

        if Y_test.ndim == 3:
            Y_test = Y_test[:, -pred_len:, :]
        elif Y_test.ndim == 2:
            Y_test = Y_test[:, -pred_len:]

        log.info(f"[{model_name}] Y 截断后: Y_train={Y_train.shape}, Y_test={Y_test.shape}")

        # ── 3D 化：统一为 (n, seq/pred, n_feat) ─────────────────
        X_train_3d = X_train.reshape(X_train.shape[0], seq_len, -1) \
            if X_train.ndim == 2 else X_train.astype(np.float32)
        Y_train_3d = Y_train.reshape(Y_train.shape[0], pred_len, -1) \
            if Y_train.ndim == 2 else Y_train.astype(np.float32)

        n_feat = X_train_3d.shape[-1]
        MAX_PREVIEW = 100

        # ── 保存前 100 条 X_test 原始值（用于波形可视化）──────────
        history_preview_raw = X_test[:MAX_PREVIEW].copy()
        log.info(f"[{model_name}] 保存 {MAX_PREVIEW} 条历史预览")

        del train_set, val_set
        gc.collect()

        # ── Dual-Dimension RevIN 训练阶段 ─────────────────────────
        X_train_norm, revin_stats_train = apply_revin(
            X_train_3d, revin_type, prefix=model_name + '_train'
        )
        Y_train_norm, _ = apply_revin(
            Y_train_3d, revin_type, prefix=model_name + '_train_Y'
        )
        log.info(f"[{model_name}] 训练数据归一化完成")

        del X_train, Y_train
        gc.collect()

        # ── 模型训练 ─────────────────────────────────────────────
        log.info(f"[{model_name}] 训练中...")
        model = ModelClass(**model_params)
        model.fit(X_train_norm, Y_train_norm)

        del X_train_3d, Y_train_3d, X_train_norm, Y_train_norm
        gc.collect()

        # ── 推理阶段 ──────────────────────────────────────────────
        X_test_3d = X_test.reshape(X_test.shape[0], seq_len, -1) \
            if X_test.ndim == 2 else X_test.astype(np.float32)
        Y_test_3d = Y_test.reshape(Y_test.shape[0], pred_len, -1) \
            if Y_test.ndim == 2 else Y_test.astype(np.float32)

        X_test_norm, revin_stats_test = apply_revin(
            X_test_3d, revin_type, prefix=model_name + '_test'
        )
        log.info(f"[{model_name}] 推理归一化完成")

        del X_test
        gc.collect()

        log.info(f"[{model_name}] 预测 {X_test_3d.shape[0]} 样本...")
        Y_pred_norm = model.predict(X_test_norm)

        del model, X_test_norm
        gc.collect()

        # ── 指标计算（在归一化空间，与 TSLib 0.3 量级对齐）────────
        metrics = calculate_all_metrics(Y_pred_norm, Y_test_3d)

        elapsed = time.time() - start_time
        log.info(f"[{model_name}] 归一化空间 MAE={metrics.get('MAE', 0):.4f} "
                 f"MSE={metrics.get('MSE', 0):.4f} elapsed={elapsed:.1f}s")

        # ── 反归一化（得到原始物理尺度）──────────────────────────
        Y_pred = inverse_revin(Y_pred_norm, revin_stats_test, revin_type)
        del Y_pred_norm
        gc.collect()

        # ── inverse_transform + 保存 .npy ─────────────────────────────
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

        # ── JSON preview（三路全部反归一化为原始尺度）─────────────
        preview_hist_flat = history_preview_raw.reshape(-1, n_feat)
        # 用 TSLib StandardScaler 反归一化 history（从原始数据尺度转回原始物理尺度）
        # 注意：history_preview_raw 本身就是原始尺度，无需再次 inverse_transform
        # 仅对展平格式做 reshape（与 preds/trues 保持一致的列表格式）
        preview_hist_list = history_preview_raw.reshape(
            MAX_PREVIEW, seq_len * n_feat
        ).tolist()
        preview_pred = Y_pred_inv[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).tolist()
        preview_true = Y_test_inv[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).tolist()

        del Y_pred, Y_pred_inv, Y_test_inv
        gc.collect()

        # ── 保存 params.json ────────────────────────────────────
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

        # ── 保存 metrics.json ────────────────────────────────────
        metrics_clean = {}
        for k, v in metrics.items():
            try:
                metrics_clean[k] = float(v) if (v == v) else 0.0
            except Exception:
                metrics_clean[k] = 0.0
        with open(os.path.join(exp_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_clean, f, indent=2)

        # ── 自动绘图（plot_comparison_samples）──────────────────
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
            log.info(f"[{model_name}] 可视化已保存: {png_path}")
        except Exception as plot_err:
            log.warning(f"[{model_name}] 绘图失败（不影响实验）: {plot_err}")

        log.info(f"[{model_name}] 实验完成 elapsed={elapsed:.1f}s")
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
        log.error(f"[{model_name}] 失败: {e}")
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
    """获取参数默认值"""
    defaults = {
        'top_k': 5,
        'weighted': True,
        'n_hash_funcs': 16,
        'n_tables': 4,
        'hamming_radius': 2,
        'candidate_cap_per_table': 256,
        'candidate_cap_total': 1024,
        'word_size': 8,
        'alphabet_size': 8,
        'epsilon_threshold': 1.0,
        'bucket_top_k': 8,
    }
    return defaults.get(param)


def _make_exp_id(model_name: str, seq_len: int, pred_len: int, config: Dict) -> str:
    """生成实验ID（含模型专属超参数后缀 + revin_type，避免同名冲突）"""
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
                + "_e" + str(config.get('epochs', 10))
                + "_k" + str(config.get('top_k', 5))
                + revin_suffix)
    elif model_name == 'RAGSearch':
        return (base
                + "_dm" + str(config.get('d_model', 32))
                + "_nh" + str(config.get('n_heads', 4))
                + "_e" + str(config.get('epochs', 10))
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

    Bug 修复 (v2.1):
    - 单个实验失败不中断整个脚本
    - 异常结果也加入 results，保证即使全部失败也能写日志
    """

    MEMORY_THRESHOLD = 0.85

    def __init__(self, args):
        self.args = args
        self.results = []
        self._memory_check()

    def _memory_check(self) -> bool:
        try:
            mem = psutil.virtual_memory()
            usage = mem.percent / 100.0
            avail_gb = mem.available / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            logger.info(f"[内存] 已用 {usage:.1%}  ({total_gb:.1f}G 总, 可用 {avail_gb:.1f}G)")

            if usage >= self.MEMORY_THRESHOLD:
                logger.warning(f"[内存] 已用 {usage:.1%} >= {self.MEMORY_THRESHOLD:.1%}，"
                               f"并行模式降级为串行")
                return False
            return True
        except Exception as e:
            logger.warning(f"[内存] 检测失败 ({e})，按保守策略串行执行")
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

            # ── PatternSearch ─────────────────────────────────────
            if m == 'PatternSearch':
                cfg.update({
                    'top_k': self.args.top_k,
                    'weighted': self.args.weighted,
                    'predict_chunk_size': getattr(self.args, 'predict_chunk_size', 2048),
                })

            # ── LSHSearch ───────────────────────────────────────
            elif m == 'LSHSearch':
                cfg.update({
                    'n_hash_funcs': self.args.n_hash_funcs,
                    'n_tables': self.args.n_tables,
                    'hamming_radius': self.args.hamming_radius,
                    'candidate_cap_per_table': self.args.candidate_cap_per_table,
                    'candidate_cap_total': self.args.candidate_cap_total,
                    'weighted': self.args.lsh_weighted,
                })

            # ── SAXSearch ───────────────────────────────────────
            elif m == 'SAXSearch':
                cfg.update({
                    'word_size': self.args.word_size,
                    'alphabet_size': self.args.alphabet_size,
                    'epsilon_threshold': self.args.epsilon_threshold,
                    'bucket_top_k': self.args.bucket_top_k,
                    'weighted': self.args.sax_weighted,
                })

            # ── DTWSearch (v3.0 新增) ───────────────────────────
            elif m == 'DTWSearch':
                cfg.update({
                    'top_k': self.args.top_k,
                    'dtw_radius': getattr(self.args, 'dtw_radius', 5),
                    'weighted': self.args.weighted,
                    'predict_chunk_size': getattr(self.args, 'predict_chunk_size', 512),
                })

            # ── MatrixProfileSearch (v3.0 新增) ─────────────────
            elif m == 'MatrixProfileSearch':
                cfg.update({
                    'top_k': self.args.top_k,
                    'subsequence_length': getattr(self.args, 'subsequence_length', None),
                    'normalize': getattr(self.args, 'mp_normalize', True),
                })

            # ── TS2VecSearch (v3.0 新增) ─────────────────────────
            elif m == 'TS2VecSearch':
                cfg.update({
                    'hidden_dim': getattr(self.args, 'hidden_dim', 64),
                    'epochs': getattr(self.args, 'ts2vec_epochs', 10),
                    'batch_size': getattr(self.args, 'ts2vec_batch_size', 128),
                    'top_k': self.args.top_k,
                    'lr': getattr(self.args, 'ts2vec_lr', 1e-3),
                    'temperature': getattr(self.args, 'temperature', 0.1),
                })

            # ── RAGSearch (v3.0 新增) ───────────────────────────
            elif m == 'RAGSearch':
                cfg.update({
                    'd_model': getattr(self.args, 'rag_d_model', 32),
                    'n_heads': getattr(self.args, 'rag_n_heads', 4),
                    'epochs': getattr(self.args, 'rag_epochs', 10),
                    'batch_size': getattr(self.args, 'rag_batch_size', 128),
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
        logger.info("时序预测基线模型实验系统  [v2.1]")
        logger.info("=" * 60)
        unique_models = len(set(c['model_name'] for c in configs))
        logger.info(f"模型: {configs[0]['model_name'] if unique_models == 1 else 'all'} ({total} 个实验)")
        logger.info("=" * 60)

        use_parallel = self.args.parallel and total > 1
        memory_safe = self._memory_check()

        if use_parallel and memory_safe:
            self._run_parallel(configs, total)
        else:
            if use_parallel and not memory_safe:
                logger.warning("并行请求被内存保护拦截，回退为串行")
            self._run_sequential(configs, total)

        self._save_log()
        return self.results

    def _run_sequential(self, configs: List[Dict], total: int):
        """串行执行，单个失败不中断"""
        logger.info(f"串行执行 {total} 个实验...\n")

        for i, cfg in enumerate(configs):
            model_name = cfg['model_name']
            logger.info(f"[{i+1}/{total}] {model_name} seq={cfg['seq_len']} pred={cfg['pred_len']}")

            # 异常隔离：失败也记录结果，继续下一个
            try:
                result = run_single_experiment(cfg)
            except Exception as e:
                logger.error(f"[{i+1}/{total}] {model_name} 抛出未捕获异常: {e}")
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
                logger.info(f"  -> 成功 MAE={m.get('MAE', 0):.4f} "
                            f"MSE={m.get('MSE', 0):.4f} elapsed={result['elapsed']:.1f}s")
            else:
                logger.warning(f"  -> 失败: {result.get('error', 'unknown')}")

    def _run_parallel(self, configs: List[Dict], total: int):
        """并行执行，单个失败不中断"""
        n_workers = min(self.args.n_workers, total, 4)
        logger.info(f"并行执行: {n_workers} workers\n")

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
                    logger.error(f"[{idx+1}/{total}] {cfg['model_name']} future 异常: {e}")
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
                    logger.info(f"  -> 成功 MAE={m.get('MAE', 0):.4f} "
                                f"MSE={m.get('MSE', 0):.4f} elapsed={result['elapsed']:.1f}s")
                else:
                    logger.warning(f"  -> 失败: {result.get('error', 'unknown')}")

    def _save_log(self):
        """保存实验日志 + summary_metrics.csv"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'dataset': self.args.data_path,
                'features': self.args.features,
                'total': len(self.results),
            },
            'experiments': []
        }

        # 汇总 CSV 行
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

            # CSV 行
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

        log_path = os.path.join(RESULTS_DIR, 'experiment_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        # 生成 summary_metrics.csv
        import pandas as pd
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_path = os.path.join(RESULTS_DIR, 'summary_metrics.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info("Summary CSV 已保存: " + csv_path)

        self._print_summary()

    def _print_summary(self):
        """打印结果摘要"""
        success = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']

        logger.info("=" * 70)
        logger.info("时序预测基线模型实验系统  [v3.0]")
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
            sep = "pred" if r['status'] == 'success' else "FAIL"
            logger.info("%-42s %-16s %-7s %-5d %-5d %-8s %s (%s)"
                        % (eid, cfg.get('model_name', ''), revin,
                           cfg.get('seq_len', 0), cfg.get('pred_len', 0),
                           mae, mse, sep))

        logger.info("-" * 70)
        logger.info("实验完成: " + str(len(success)) + "/" + str(len(self.results))
                    + " 成功 " + str(len(failed)) + " 失败")
        logger.info("日志: " + os.path.join(RESULTS_DIR, 'experiment_log.json'))
        logger.info("汇总: " + os.path.join(RESULTS_DIR, 'summary_metrics.csv'))


# ============================================================
# 仪表盘启动
# ============================================================

def launch_dashboard():
    """启动 Streamlit 仪表盘（始终成功启动）"""
    import subprocess
    import webbrowser

    dashboard_path = os.path.join(PROJECT_ROOT, 'dashboard', 'app.py')

    if not os.path.exists(dashboard_path):
        logger.error(f"仪表盘文件不存在: {dashboard_path}")
        return

    cmd = [
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.port', '8501', '--server.headless', 'true',
        '--server.address', '0.0.0.0'  # 允许外部网络(Ingress/NodePort)访问
    ]
    try:
        subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        logger.info("仪表盘进程已启动: http://localhost:8501")
        time.sleep(2)
        webbrowser.open('http://localhost:8501')
        logger.info("已在默认浏览器中打开仪表盘（若失败请手动访问上述地址）")
    except Exception as e:
        logger.error(f"启动仪表盘失败: {e}")
        logger.info(f"请手动运行: {' '.join(cmd)}")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='时序预测基线模型')

    # 模式选择
    parser.add_argument('--model', type=str, default='PatternSearch',
                       help='模型: PatternSearch, LSHSearch, SAXSearch, DTWSearch, '
                            'MatrixProfileSearch, TS2VecSearch, RAGSearch, all')
    parser.add_argument('--dashboard', action='store_true',
                       help='运行后启动可视化仪表盘')
    parser.add_argument('--skip_run', action='store_true',
                       help='跳过实验，仅启动仪表盘')

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

    # ── v3.0 新增模型参数 ─────────────────────────────────────────

    # 通用参数（跨模型共享）
    parser.add_argument('--predict_chunk_size', type=int, default=512,
                       help='预测时分块大小（DTWSearch）')

    # DTWSearch
    parser.add_argument('--dtw_radius', type=int, default=5,
                       help='DTW Sakoe-Chiba 约束窗口半径，默认 5')

    # MatrixProfileSearch
    parser.add_argument('--subsequence_length', type=int, default=None,
                       help='MatrixProfile 子序列长度（None=seq_len）')
    parser.add_argument('--mp_normalize', type=lambda x: x.lower() == 'true', default=True,
                       help='MatrixProfile 是否 z-normalize')

    # TS2VecSearch
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='TS2Vec / RAG 隐向量维度，默认 64')
    parser.add_argument('--ts2vec_epochs', type=int, default=10,
                       help='TS2Vec 对比学习训练轮数，默认 10')
    parser.add_argument('--ts2vec_batch_size', type=int, default=128,
                       help='TS2Vec 训练批大小，默认 128')
    parser.add_argument('--ts2vec_lr', type=float, default=1e-3,
                       help='TS2Vec 学习率，默认 1e-3')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='对比损失温度，默认 0.1')

    # RAGSearch
    parser.add_argument('--rag_d_model', type=int, default=32,
                       help='RAG Cross-Attention 隐向量维度，默认 32')
    parser.add_argument('--rag_n_heads', type=int, default=4,
                       help='RAG 注意力头数，默认 4')
    parser.add_argument('--rag_epochs', type=int, default=10,
                       help='RAG 训练轮数，默认 10')
    parser.add_argument('--rag_batch_size', type=int, default=128,
                       help='RAG 训练批大小，默认 128')
    parser.add_argument('--rag_lr', type=float, default=1e-3,
                       help='RAG 学习率，默认 1e-3')
    parser.add_argument('--rag_weight_decay', type=float, default=1e-4,
                       help='RAG 权重衰减，默认 1e-4')

    # 执行参数
    parser.add_argument('--parallel', action='store_true',
                       help='启用并行计算（内存 > 85%% 时自动降级）')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='并行进程数（最大 4）')
    parser.add_argument('--use_gpu', action='store_true',
                       help='若 torch.cuda 可用则在 GPU 上做张量距离/投影（PatternSearch/LSH/SAX）')
    parser.add_argument('--revin_type', type=str, default='none',
                       choices=['none', 'temporal', 'feature', 'dual'],
                       help='归一化类型：none=无归一化, temporal=时间维度 InstanceNorm, '
                            'feature=特征维度 ChannelNorm, dual=先 feature 再 temporal')

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Bug 修复 (v2.1): 优先级 skip_run > dashboard ──
    if args.skip_run:
        launch_dashboard()
        sys.exit(0)

    # 运行实验（异常隔离，即使全部失败也继续）
    runner = ExperimentRunner(args)
    runner.run()

    # ── Bug 修复 (v2.1): dashboard 必定启动 ──
    if args.dashboard:
        launch_dashboard()


if __name__ == '__main__':
    main()
