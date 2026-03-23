"""
统一入口脚本 - 时序预测基线模型系统

Usage:
    # 单模型运行
    python run.py --model PatternSearch

    # 所有模型
    python run.py --model all

    # 并行运行 + 启动仪表盘
    python run.py --model all --parallel --dashboard

    # 自定义参数网格
    python run.py --model all --seq_len 96 192 --pred_len 24 48 96

    # 仅启动仪表盘
    python run.py --dashboard --skip_run
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count

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
        'class': None,  # 延迟导入
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

    Args:
        config: 实验配置字典

    Returns:
        实验结果字典
    """
    import numpy as np
    from data_provider.data_loader import get_data, get_X_Y_from_dataset
    from utils.metrics import calculate_all_metrics

    start_time = time.time()
    model_name = config['model_name']
    seq_len = config['seq_len']
    pred_len = config['pred_len']

    try:
        # 导入模型
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

        # 加载数据
        train_set, val_set, test_set = get_data(args)
        X_train, Y_train = get_X_Y_from_dataset(train_set)
        X_test, Y_test = get_X_Y_from_dataset(test_set)

        # 创建并训练模型
        model = ModelClass(**model_params)
        model.fit(X_train, Y_train)

        # 预测
        Y_pred = model.predict(X_test)
        if Y_pred.ndim == 3:
            Y_pred = Y_pred[:, :, 0]  # 取第一个特征
        if Y_test.ndim == 3:
            Y_test = Y_test[:, :, 0]

        # 反归一化
        Y_pred_orig = test_set.inverse_transform(Y_pred.reshape(-1, 1)).reshape(Y_pred.shape)
        Y_test_orig = test_set.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)

        # 计算指标
        metrics = calculate_all_metrics(Y_pred_orig, Y_test_orig)

        # 保存预测结果
        exp_id = _make_exp_id(model_name, seq_len, pred_len, config)
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_preds.npy"), Y_pred_orig[:100])
        np.save(os.path.join(RESULTS_DIR, f"{exp_id}_trues.npy"), Y_test_orig[:100])

        elapsed = time.time() - start_time

        return {
            'config': config,
            'metrics': metrics,
            'status': 'success',
            'elapsed': elapsed
        }

    except Exception as e:
        import traceback
        return {
            'config': config,
            'metrics': {},
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed': time.time() - start_time
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
    """实验运行器（可导入复用）"""

    def __init__(self, args):
        self.args = args
        self.results = []

    def build_configs(self) -> List[Dict[str, Any]]:
        """构建实验配置列表"""
        models = []
        if self.args.model == 'all':
            models = list(MODEL_REGISTRY.keys())
        else:
            models = [m.strip() for m in self.args.model.split(',')]

        seq_lens = self.args.seq_len if self.args.seq_len else [96]
        pred_lens = self.args.pred_len if self.args.pred_len else [48]

        # 基础配置模板
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

        # 添加全局配置
        for cfg in configs:
            cfg['root_path'] = self.args.root_path
            cfg['data_path'] = self.args.data_path
            cfg['features'] = self.args.features
            cfg['target'] = self.args.target

        return configs

    def run(self):
        """运行实验"""
        configs = self.build_configs()
        total = len(configs)

        print(f"\n{'='*60}")
        print(f"时序预测基线模型实验系统")
        print(f"{'='*60}")
        print(f"模型: {configs[0]['model_name'] if len(set(c['model_name'] for c in configs)) == 1 else 'all'}")
        print(f"实验数: {total}")
        print(f"并行: {self.args.parallel}")
        print(f"{'='*60}\n")

        if self.args.parallel and total > 1:
            # 并行执行
            n_workers = min(self.args.n_workers, cpu_count(), total)
            print(f"使用 {n_workers} 个进程并行执行...")

            with Pool(n_workers) as pool:
                self.results = pool.map(run_single_experiment, configs)
        else:
            # 串行执行
            for i, cfg in enumerate(configs):
                print(f"[{i+1}/{total}] {cfg['model_name']} seq={cfg['seq_len']} pred={cfg['pred_len']}")
                result = run_single_experiment(cfg)
                self.results.append(result)

        self._save_log()

        return self.results

    def _save_log(self):
        """保存实验日志"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'dataset': self.args.data_path,
                'features': self.args.features,
                'total': len(self.results),
            },
            'experiments': self.results
        }

        log_path = os.path.join(RESULTS_DIR, 'experiment_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        # 打印摘要
        self._print_summary()

    def _print_summary(self):
        """打印结果摘要"""
        success = [r for r in self.results if r['status'] == 'success']

        print(f"\n{'='*60}")
        print(f"实验完成: {len(success)}/{len(self.results)} 成功")
        print(f"{'='*60}")

        # 按模型和预测长度汇总
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

    # 启动服务
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
                       help='启用并行计算')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='并行进程数')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip_run:
        # 仅启动仪表盘
        launch_dashboard()
    else:
        # 运行实验
        runner = ExperimentRunner(args)
        runner.run()

        if args.dashboard:
            launch_dashboard()


if __name__ == '__main__':
    main()
