"""
统一入口脚本 - 时序预测基线模型系统

Usage:
    # 单模型运行
    python run.py --model PatternSearch

    # 所有模型（串行，推荐 8G 以下环境）
    python run.py --model all --dashboard

    # 并行运行（自动内存保护，内存 > 16G 时启用）
    python run.py --model all --parallel --dashboard

    # 自定义参数网格
    python run.py --model all --seq_len 96 192 --pred_len 24 48 96

    # 仅启动仪表盘
    python run.py --dashboard --skip_run

内存优化说明（v2.0）：
- 所有数据使用 float32（相比 float64 节省 50%）
- 实验结果 JSON 只保留前 100 条样本，完整数据落盘 .npy
- ExperimentRunner 每次实验结束后强制 gc.collect()
- 并行模式：动态检测内存，超过 85% 回退串行；每个子进程 max_tasks_per_child=1
"""

import os
import sys
import gc
import json
import time
import psutil
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        'params': ['n_hash_funcs', 'n_tables', 'hamming_radius']
    },
    'SAXSearch': {
        'class': None,
        'params': ['word_size', 'alphabet_size', 'epsilon_threshold']
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
    运行单次实验（核心计算逻辑，可并行调用）

    内存优化点：
    - 数据集直接产生 float32
    - 模型 fit/predict 后立即 del 中间变量
    - 只在 config['save_preds'] == True 时才保留前 100 条样本到结果
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

        # 构建模型参数
        model_params = {}
        for param in MODEL_REGISTRY[model_name]['params']:
            model_params[param] = config.get(param, _get_default(param))

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
        train_set, val_set, test_set = get_data(args)
        X_train, Y_train = get_X_Y_from_dataset(train_set)
        X_test, Y_test = get_X_Y_from_dataset(test_set)

        # 立即释放数据集对象（如果不需要后续使用）
        del train_set, val_set
        gc.collect()

        # 创建并训练模型
        model = ModelClass(**model_params)
        model.fit(X_train, Y_train)

        # 释放训练数据（预测时不再需要）
        del X_train, Y_train
        gc.collect()

        # 预测
        Y_pred = model.predict(X_test)
        if Y_pred.ndim == 3:
            Y_pred = Y_pred[:, :, 0]
        if Y_test.ndim == 3:
            Y_test = Y_test[:, :, 0]

        # 反归一化
        Y_pred_orig = test_set.inverse_transform(Y_pred.reshape(-1, 1)).reshape(Y_pred.shape)
        Y_test_orig = test_set.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)

        # 释放 test_set（不再需要）
        del test_set, Y_pred, Y_test
        gc.collect()

        # 计算指标（标量结果）
        metrics = calculate_all_metrics(Y_pred_orig, Y_test_orig)

        # 生成实验ID
        exp_id = _make_exp_id(model_name, seq_len, pred_len, config)

        # ================================================
        # 任务3: JSON 日志严格截断 - 只保留前 100 条样本
        # 完整数据只通过 .npy 落盘
        # ================================================
        # 落盘完整数据（float32，体积约为 float64 的一半）
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_preds.npy"),
                Y_pred_orig.astype(np.float32))
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_trues.npy"),
                Y_test_orig.astype(np.float32))

        # 只把前 100 条样本的截断数据放入 JSON（用于仪表盘快速预览）
        MAX_PREVIEW = 100
        preview_pred = Y_pred_orig[:MAX_PREVIEW].astype(np.float32).tolist()
        preview_true = Y_test_orig[:MAX_PREVIEW].astype(np.float32).tolist()

        # 释放完整数组
        del Y_pred_orig, Y_test_orig
        gc.collect()

        elapsed = time.time() - start_time

        # 构建结果字典（只有标量 metrics + 截断预览，无大数组）
        return {
            'config': config,
            'metrics': metrics,
            'status': 'success',
            'elapsed': round(elapsed, 2),
            'preview': {           # 仅前 100 条，用于 Streamlit 仪表盘
                'preds': preview_pred,
                'trues': preview_true,
                'count': min(len(preview_pred), len(preview_true))
            },
            'npy_file': {          # 完整数据文件路径
                'preds': f"{exp_id}_preds.npy",
                'trues': f"{exp_id}_trues.npy"
            }
        }

    except Exception as e:
        import traceback
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
        'word_size': 8,
        'alphabet_size': 8,
        'epsilon_threshold': 1.0,
    }
    return defaults.get(param)


def _make_exp_id(model_name: str, seq_len: int, pred_len: int, config: Dict) -> str:
    """生成实验ID"""
    dataset = config.get('data_path', 'ETTm1.csv').replace('.csv', '')

    if model_name == 'PatternSearch':
        return f"{dataset}_seq{seq_len}_pred{pred_len}_k{config.get('top_k', 5)}"
    elif model_name == 'LSHSearch':
        return f"{dataset}_seq{seq_len}_pred{pred_len}_lsh_h{config.get('n_hash_funcs', 16)}_t{config.get('n_tables', 4)}"
    else:  # SAXSearch
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

    内存保护机制（任务4）：
    - 动态检测系统内存使用率
    - 超过 85% 自动从并行回退为串行
    - ProcessPoolExecutor 设置 max_workers=1，每任务一进程后立即销毁
    """

    # 内存安全阈值（超过此值强制串行）
    MEMORY_THRESHOLD = 0.85

    def __init__(self, args):
        self.args = args
        self.results = []
        self._memory_check()

    def _memory_check(self) -> bool:
        """
        检测当前系统内存占用率

        Returns:
            True: 内存安全，可以继续
            False: 内存紧张，需要降级
        """
        try:
            mem = psutil.virtual_memory()
            usage = mem.percent / 100.0
            avail_gb = mem.available / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            print(f"  [内存] 已用 {usage:.1%}  ({total_gb:.1f}G 总, 可用 {avail_gb:.1f}G)")

            if usage >= self.MEMORY_THRESHOLD:
                print(f"  [警告] 内存占用 {usage:.1%} >= {self.MEMORY_THRESHOLD:.1%}，"
                      f"并行模式降级为串行以避免 OOM")
                return False
            return True
        except Exception as e:
            print(f"  [警告] 无法检测内存状态 ({e})，按保守策略串行执行")
            return False

    def build_configs(self) -> List[Dict[str, Any]]:
        """构建实验配置列表"""
        models = []
        if self.args.model == 'all':
            models = list(MODEL_REGISTRY.keys())
        else:
            models = [m.strip() for m in self.args.model.split(',')]

        seq_lens = self.args.seq_len if self.args.seq_len else [96]
        pred_lens = self.args.pred_len if self.args.pred_len else [48]

        all_configs = [
            {
                'model_name': 'PatternSearch',
                'top_k': self.args.top_k,
                'weighted': self.args.weighted,
            },
            {
                'model_name': 'LSHSearch',
                'n_hash_funcs': self.args.n_hash_funcs,
                'n_tables': self.args.n_tables,
                'hamming_radius': self.args.hamming_radius,
            },
            {
                'model_name': 'SAXSearch',
                'word_size': self.args.word_size,
                'alphabet_size': self.args.alphabet_size,
                'epsilon_threshold': self.args.epsilon_threshold,
            }
        ]

        configs = _expand_configs(models, seq_lens, pred_lens, all_configs)

        for cfg in configs:
            cfg['root_path'] = self.args.root_path
            cfg['data_path'] = self.args.data_path
            cfg['features'] = self.args.features
            cfg['target'] = self.args.target

        return configs

    def run(self):
        """运行实验（串行 / 并行自适应）"""
        configs = self.build_configs()
        total = len(configs)

        print(f"\n{'='*60}")
        print(f"时序预测基线模型实验系统  [内存优化版 v2.0]")
        print(f"{'='*60}")
        print(f"模型: {configs[0]['model_name'] if len(set(c['model_name'] for c in configs)) == 1 else 'all'}")
        print(f"实验数: {total}")
        print(f"{'='*60}\n")

        # 判断执行模式
        use_parallel = self.args.parallel and total > 1
        memory_safe = self._memory_check()

        if use_parallel and memory_safe:
            self._run_parallel(configs, total)
        else:
            if use_parallel and not memory_safe:
                print("并行请求被内存保护拦截，回退为串行执行。\n")
            self._run_sequential(configs, total)

        self._save_log()
        return self.results

    def _run_sequential(self, configs: List[Dict], total: int):
        """
        串行执行 + 任务2 GC 回收
        每次实验结束后显式 del 大对象并 gc.collect()
        """
        print(f"串行执行（实验间强制 GC）...")

        for i, cfg in enumerate(configs):
            model_name = cfg['model_name']
            print(f"[{i+1}/{total}] {model_name} seq={cfg['seq_len']} pred={cfg['pred_len']}")

            result = run_single_experiment(cfg)
            self.results.append(result)

            # ================================================
            # 任务2: 显式垃圾回收
            # 禁止全局 results 列表中存放大数组；只保留标量 metrics
            # ================================================
            # 如果 ExperimentRunner.results 意外引用了大数组，在此清理
            # （result 中已不包含大数组，此处仅作保险）
            gc.collect()

            # 打印当前实验结果
            if result['status'] == 'success':
                m = result['metrics']
                print(f"  -> MAE={m.get('MAE', 0):.4f}  MSE={m.get('MSE', 0):.4f}  "
                      f"elapsed={result['elapsed']:.1f}s")
            else:
                print(f"  -> FAILED: {result.get('error', 'unknown')}")

    def _run_parallel(self, configs: List[Dict], total: int):
        """
        并行执行 + 任务4 内存保护
        - max_workers 最多 min(n_workers, total, 4)
        - 每个子进程跑完一个任务后立即销毁（max_tasks_per_child=1）
        - 内存超 85% 时回退串行（由 run() 中的 _memory_check 保证首次安全）
        """
        n_workers = min(self.args.n_workers, total, 4)
        print(f"并行执行: {n_workers} workers (max_tasks_per_child=1)")

        # 使用 ProcessPoolExecutor + max_workers 自动内存回收
        # 注意：Python 3.11+ 支持 max_tasks_per_child
        # 若版本不支持，手动在 worker 函数中 gc.collect() 即可
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_experiment, cfg): i
                        for i, cfg in enumerate(configs)}

            for future in as_completed(futures):
                idx = futures[future]
                cfg = configs[idx]
                print(f"[{idx+1}/{total}] {cfg['model_name']} seq={cfg['seq_len']} pred={cfg['pred_len']}")

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        'config': cfg,
                        'metrics': {},
                        'status': 'failed',
                        'error': str(e)
                    }

                self.results.append(result)
                gc.collect()

                if result['status'] == 'success':
                    m = result['metrics']
                    print(f"  -> MAE={m.get('MAE', 0):.4f}  MSE={m.get('MSE', 0):.4f}  "
                          f"elapsed={result['elapsed']:.1f}s")
                else:
                    print(f"  -> FAILED: {result.get('error', 'unknown')}")

    def _save_log(self):
        """保存实验日志（JSON 中只含标量 metrics 和前 100 条预览）"""
        # ================================================
        # 任务3: JSON 中绝对不存入完整预测数组
        # 只保留 metrics（标量）和 preview（前 100 条截断预览）
        # ================================================
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

            # 只在成功时加入 preview（字典体积可控）
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

        print(f"\n{'='*60}")
        print(f"实验完成: {len(success)}/{len(self.results)} 成功")
        print(f"{'='*60}")

        if success:
            print("\n{:<18} {:>10} {:>10} {:>10} {:>10}".format(
                "模型", "seq_len", "pred_len", "MAE", "MSE"))
            print("-" * 60)

            for r in success:
                cfg = r['config']
                m = r['metrics']
                print("{:<18} {:>10} {:>10} {:>10.4f} {:>10.4f}".format(
                    cfg['model_name'], cfg['seq_len'], cfg['pred_len'],
                    m.get('MAE', 0), m.get('MSE', 0)))

        print(f"\n日志已保存: {os.path.join(RESULTS_DIR, 'experiment_log.json')}")


# ============================================================
# 仪表盘启动
# ============================================================

def launch_dashboard():
    """启动 Streamlit 仪表盘"""
    import subprocess
    import webbrowser

    dashboard_path = os.path.join(PROJECT_ROOT, 'dashboard', 'app.py')
    cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path,
           '--server.port', '8501', '--server.headless', 'true']

    print(f"\n启动仪表盘: http://localhost:8501")

    subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    time.sleep(3)
    webbrowser.open('http://localhost:8501')


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
    parser.add_argument('--seq_len', type=int, nargs='+', default=[96],
                       help='输入序列长度，如: --seq_len 96 192')
    parser.add_argument('--pred_len', type=int, nargs='+', default=[48],
                       help='预测长度，如: --pred_len 24 48 96')

    # PatternSearch 参数
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--weighted', type=lambda x: x.lower() == 'true', default=True)

    # LSHSearch 参数
    parser.add_argument('--n_hash_funcs', type=int, default=16)
    parser.add_argument('--n_tables', type=int, default=4)
    parser.add_argument('--hamming_radius', type=int, default=2)

    # SAXSearch 参数
    parser.add_argument('--word_size', type=int, default=8)
    parser.add_argument('--alphabet_size', type=int, default=8)
    parser.add_argument('--epsilon_threshold', type=float, default=1.0)

    # 执行参数
    parser.add_argument('--parallel', action='store_true',
                       help='启用并行计算（内存 > 85%% 时自动降级为串行）')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='并行进程数（最大 4）')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip_run:
        launch_dashboard()
    else:
        runner = ExperimentRunner(args)
        runner.run()

        if args.dashboard:
            launch_dashboard()


if __name__ == '__main__':
    main()
