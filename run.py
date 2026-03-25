"""
统一入口脚本 - 时序预测基线模型系统

Bug 修复 (v2.1):
- 移除 Y_pred[:, :, 0] / Y_test[:, :, 0] 破坏性切片，解决多变量 broadcast 错误
- 引入 logging 模块（INFO级别 + 时间戳）
- 异常隔离：单个模型失败不影响后续实验
- Dashboard：无论实验结果如何，只要带 --dashboard 必定启动
- skip_run + dashboard：直接启动仪表盘后 exit(0)

Usage:
    python run.py --model all --dashboard              # 所有模型 + 仪表盘
    python run.py --model PatternSearch --dashboard    # 单模型 + 仪表盘
    python run.py --dashboard --skip_run               # 仅启动仪表盘
"""

import os
import sys
import gc
import json
import time
import logging
import psutil
import argparse
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# 全局日志配置（带时间戳）
# ============================================================

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

# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY = {
    'PatternSearch': {
        'class': None,
        'params': ['top_k', 'weighted']
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
    }
}


def import_models():
    """延迟导入模型类"""
    from models.PatternSearch import PatternSearch
    from models.LSHSearch import LSHSearch
    from models.SAXSearch import SAXSearch
    MODEL_REGISTRY['PatternSearch']['class'] = PatternSearch
    MODEL_REGISTRY['LSHSearch']['class'] = LSHSearch
    MODEL_REGISTRY['SAXSearch']['class'] = SAXSearch


# ============================================================
# 核心计算逻辑（可独立复用）
# ============================================================

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单次实验

    Bug 修复 (v2.1):
    - 不再对 Y_pred / Y_test 强制做 [:, :, 0]，保留多变量维度
    - inverse_transform 根据实际 n_features 动态处理 2D/3D 形状
    """
    import numpy as np
    from data_provider.data_loader import get_data, get_X_Y_from_dataset
    from utils.metrics import calculate_all_metrics

    start_time = time.time()
    model_name = config['model_name']
    seq_len = config['seq_len']
    pred_len = config['pred_len']

    try:
        import_models()
        ModelClass = MODEL_REGISTRY[model_name]['class']

        # 构建模型参数（全部通过 kwargs 安全传递）
        model_params = {}
        for param in MODEL_REGISTRY[model_name]['params']:
            model_params[param] = config.get(param, _get_default(param))

        import torch
        use_gpu = bool(config.get('use_gpu', False))
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        if use_gpu and device == 'cpu':
            logger.warning(f"[{model_name}] 请求 GPU 但 torch.cuda 不可用，已回退 CPU")
        model_params['device'] = device
        logger.info(f"[{model_name}] 计算设备: {device}")

        # 创建参数对象
        class Args:
            pass
        args = Args()
        args.root_path = config.get('root_path', './ETT_data')
        args.data_path = config.get('data_path', 'ETTm1.csv')
        args.seq_len = seq_len
        args.pred_len = pred_len
        args.features = config.get('features', 'M')
        args.target = config.get('target', 'OT')

        # 加载数据（float32）
        logger.info(f"[{model_name}] 加载数据 seq={seq_len} pred={pred_len}")
        train_set, val_set, test_set = get_data(args)
        X_train, Y_train = get_X_Y_from_dataset(train_set)
        X_test, Y_test = get_X_Y_from_dataset(test_set)

        logger.info(f"[{model_name}] 数据就绪: X_train={X_train.shape}, X_test={X_test.shape}")

        # 释放数据集对象
        del train_set, val_set
        gc.collect()

        # 创建并训练模型
        logger.info(f"[{model_name}] 训练...")
        model = ModelClass(**model_params)
        model.fit(X_train, Y_train)

        # 释放训练数据
        del X_train, Y_train
        gc.collect()

        # ─────────────────────────────────────────────────────────────────
        # 任务5: DLinear-style Instance-level 去均值 (Mean-Shift)
        # 对每个测试序列减去其特征维度的均值，提升检索稳定性
        # ─────────────────────────────────────────────────────────────────
        seq_mean = np.mean(X_test, axis=1, keepdims=True)          # (n_test, 1, n_feat)
        X_test_centered = (X_test - seq_mean).astype(np.float32)    # (n_test, seq_len, n_feat)

        # 预测（用中心化后的序列）
        logger.info(f"[{model_name}] 预测 {X_test.shape[0]} 样本 (mean-shift 模式)...")
        Y_pred_centered = model.predict(X_test_centered)             # (n_test, pred_len, n_feat)

        # 释放模型和中心化输入
        del model, X_test_centered
        gc.collect()

        # 将均值加回（恢复到原始尺度预测）
        # Y_pred_centered 是差值形式，加上对应样本的 seq_mean 即为预测值
        Y_pred = Y_pred_centered + seq_mean
        del Y_pred_centered, seq_mean
        gc.collect()

        # ─────────────────────────────────────────────────────────────────
        # 任务7: 历史上下文数据落盘（同样做 mean-shift 保持一致）
        # 保存 X_test（原始）用于 dashboard 历史语境展示
        # ─────────────────────────────────────────────────────────────────
        exp_id = _make_exp_id(model_name, seq_len, pred_len, config)
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_X_test.npy"),
                X_test.astype(np.float32))
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_preds.npy"),
                Y_pred.astype(np.float32))
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_trues.npy"),
                Y_test.astype(np.float32))

        # ─────────────────────────────────────────────────
        # 反归一化：按“最安全模板”整段替换（避免局部变量作用域/分支风险）
        # ─────────────────────────────────────────────────
        n_features = Y_test.shape[-1] if Y_test.ndim == 3 else 1
        n_test = Y_pred.shape[0]
        p_len = Y_pred.shape[1]

        # 安全展平 (样本数 * 预测长度, 特征数)
        Y_pred_flat = Y_pred.reshape(-1, n_features)
        Y_test_flat = Y_test.reshape(-1, n_features)

        # 反归一化
        Y_pred_orig = test_set.inverse_transform(Y_pred_flat)
        Y_test_orig = test_set.inverse_transform(Y_test_flat)

        # 安全恢复维度
        if n_features > 1:
            Y_pred_orig = Y_pred_orig.reshape(n_test, p_len, n_features)
            Y_test_orig = Y_test_orig.reshape(n_test, p_len, n_features)
        else:
            Y_pred_orig = Y_pred_orig.reshape(n_test, p_len)
            Y_test_orig = Y_test_orig.reshape(n_test, p_len)

        del test_set, Y_pred_flat, Y_test_flat, Y_pred, Y_test
        gc.collect()

        # 计算指标
        metrics = calculate_all_metrics(Y_pred_orig, Y_test_orig)

        # 生成实验ID
        exp_id = _make_exp_id(model_name, seq_len, pred_len, config)

        # 落盘完整数据
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_preds.npy"),
                Y_pred_orig.astype(np.float32))
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_trues.npy"),
                Y_test_orig.astype(np.float32))

        # JSON 只保留前 100 条预览
        MAX_PREVIEW = 100
        if Y_pred_orig.ndim == 3:
            preview_pred = Y_pred_orig[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).astype(np.float32).tolist()
            preview_true = Y_test_orig[:MAX_PREVIEW].reshape(MAX_PREVIEW, -1).astype(np.float32).tolist()
        else:
            preview_pred = Y_pred_orig[:MAX_PREVIEW].astype(np.float32).tolist()
            preview_true = Y_test_orig[:MAX_PREVIEW].astype(np.float32).tolist()

        elapsed = time.time() - start_time

        logger.info(f"[{model_name}] 成功 MAE={metrics.get('MAE', 0):.4f} "
                    f"MSE={metrics.get('MSE', 0):.4f} elapsed={elapsed:.1f}s")

        # 释放完整数组
        del Y_pred_orig, Y_test_orig
        gc.collect()

        return {
            'config': config,
            'metrics': metrics,
            'status': 'success',
            'elapsed': round(elapsed, 2),
            'preview': {
                'preds': preview_pred,
                'trues': preview_true,
                'count': min(len(preview_pred), len(preview_true))
            },
            'npy_file': {
                'preds': f"{exp_id}_preds.npy",
                'trues': f"{exp_id}_trues.npy",
                'x_test': f"{exp_id}_X_test.npy",
            }
        }

    except Exception as e:
        import traceback
        logger.error(f"[{model_name}] 失败: {e}")
        logger.debug(traceback.format_exc())
        return {
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
    """生成实验ID"""
    dataset = config.get('data_path', 'ETTm1.csv').replace('.csv', '')

    if model_name == 'PatternSearch':
        return f"{dataset}_seq{seq_len}_pred{pred_len}_k{config.get('top_k', 5)}"
    elif model_name == 'LSHSearch':
        return f"{dataset}_seq{seq_len}_pred{pred_len}_lsh_h{config.get('n_hash_funcs', 16)}_t{config.get('n_tables', 4)}"
    else:
        return f"{dataset}_seq{seq_len}_pred{pred_len}_sax_w{config.get('word_size', 8)}_a{config.get('alphabet_size', 8)}"


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

        all_configs = [
            {'model_name': 'PatternSearch', 'top_k': self.args.top_k, 'weighted': self.args.weighted},
            {'model_name': 'LSHSearch', 'n_hash_funcs': self.args.n_hash_funcs,
             'n_tables': self.args.n_tables, 'hamming_radius': self.args.hamming_radius,
             'candidate_cap_per_table': self.args.candidate_cap_per_table,
             'candidate_cap_total': self.args.candidate_cap_total,
             'weighted': self.args.lsh_weighted},
            {'model_name': 'SAXSearch', 'word_size': self.args.word_size,
             'alphabet_size': self.args.alphabet_size, 'epsilon_threshold': self.args.epsilon_threshold,
             'bucket_top_k': self.args.bucket_top_k, 'weighted': self.args.sax_weighted},
        ]

        configs = _expand_configs(models, seq_lens, pred_lens, all_configs)

        for cfg in configs:
            cfg['root_path'] = self.args.root_path
            cfg['data_path'] = self.args.data_path
            cfg['features'] = self.args.features
            cfg['target'] = self.args.target
            cfg['use_gpu'] = getattr(self.args, 'use_gpu', False)

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
        """保存实验日志"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'dataset': self.args.data_path,
                'features': self.args.features,
                'total': len(self.results),
            },
            'experiments': []
        }

        for r in self.results:
            exp_entry = {
                'config': r['config'],
                'metrics': r.get('metrics', {}),
                'status': r['status'],
                'elapsed': r.get('elapsed', 0),
            }

            if r['status'] == 'success' and 'preview' in r:
                exp_entry['preview'] = r['preview']
                exp_entry['npy_file'] = r.get('npy_file', {})

            if r['status'] == 'failed':
                exp_entry['error'] = r.get('error', '')

            log['experiments'].append(exp_entry)

        log_path = os.path.join(RESULTS_DIR, 'experiment_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        self._print_summary()

    def _print_summary(self):
        """打印结果摘要"""
        success = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']

        logger.info("=" * 60)
        logger.info(f"实验完成: {len(success)}/{len(self.results)} 成功 {len(failed)} 失败")
        logger.info("=" * 60)

        if success:
            logger.info("{:<18} {:>10} {:>10} {:>10} {:>10}".format(
                "模型", "seq_len", "pred_len", "MAE", "MSE"))
            logger.info("-" * 60)
            for r in success:
                cfg = r['config']
                m = r['metrics']
                logger.info("{:<18} {:>10} {:>10} {:>10.4f} {:>10.4f}".format(
                    cfg['model_name'], cfg['seq_len'], cfg['pred_len'],
                    m.get('MAE', 0), m.get('MSE', 0)))

        if failed:
            logger.warning("失败实验:")
            for r in failed:
                cfg = r['config']
                logger.warning(f"  - {cfg['model_name']} seq={cfg['seq_len']} "
                               f"pred={cfg['pred_len']}: {r.get('error', '')}")

        logger.info(f"\n日志已保存: {os.path.join(RESULTS_DIR, 'experiment_log.json')}")


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
                       help='模型: PatternSearch, LSHSearch, SAXSearch, all')
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

    # 执行参数
    parser.add_argument('--parallel', action='store_true',
                       help='启用并行计算（内存 > 85%% 时自动降级）')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='并行进程数（最大 4）')
    parser.add_argument('--use_gpu', action='store_true',
                       help='若 torch.cuda 可用则在 GPU 上做张量距离/投影（PatternSearch/LSH/SAX）')

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
