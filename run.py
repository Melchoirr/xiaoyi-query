"""
统一的命令行入口脚本
提供 argparse 参数化运行接口
"""

import argparse
import os
import sys


def parse_args():
    """
    解析命令行参数

    Returns:
        args: 参数字典
    """
    parser = argparse.ArgumentParser(
        description='PatternSearch: Memory-based Time Series Forecasting Baseline'
    )

    # 数据参数
    parser.add_argument('--root_path', type=str, default='./ETT_data',
                        help='数据根目录路径')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv',
                        help='数据文件名')
    parser.add_argument('--target', type=str, default='OT',
                        help='目标列名（用于单变量模式）')

    # 模型参数
    parser.add_argument('--top_k', type=int, default=5,
                        help='近邻数量k')
    parser.add_argument('--weighted', action='store_true', default=True,
                        help='使用逆距离加权平均（默认开启）')
    parser.add_argument('--no_weighted', dest='weighted', action='store_false',
                        help='禁用逆距离加权，使用简单平均')

    # 序列长度参数
    parser.add_argument('--seq_len', type=int, default=96,
                        help='输入序列长度（历史窗口）')
    parser.add_argument('--pred_len', type=int, default=48,
                        help='预测序列长度（未来窗口）')

    # 特征模式
    parser.add_argument('--features', type=str, default='S',
                        choices=['M', 'S'],
                        help='M: 多变量预测多变量, S: 单变量预测单变量')

    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（用于可重复性）')

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """
    设置随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置输出目录
    args.output_dir = args.output_dir or './results'

    # 动态导入并运行实验
    print("=" * 60)
    print("PatternSearch: Memory-based Time Series Forecasting")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dataset: {args.data_path}")
    print(f"  Target: {args.target}")
    print(f"  Features: {args.features}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  pred_len: {args.pred_len}")
    print(f"  top_k: {args.top_k}")
    print(f"  weighted: {args.weighted}")
    print(f"  random_seed: {args.seed}")
    print("=" * 60)

    try:
        from exp.exp_search import Exp_Search

        exp = Exp_Search(args)
        metrics = exp.test(save_results=True)

        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
