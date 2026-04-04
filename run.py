"""
run.py - 统一时序预测框架 (支持7种检索模型)

用法示例:
    python run.py --model PatternSearch --data ETTh1 --seq_len 512 --pred_len 96
    python run.py --model LSHSearch --data ETTh1 --seq_len 512 --pred_len 96
"""

import argparse
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting (Retrieval Models)')

    # ========================
    # 模型选择（7种检索模型）
    # ========================
    parser.add_argument('--model', type=str, default='PatternSearch',
                        choices=[
                            'PatternSearch', 'LSHSearch', 'SAXSearch', 'DTWSearch',
                            'MatrixProfileSearch', 'TS2VecSearch', 'RAGSearch'
                        ],
                        help='模型选择')

    # ========================
    # 数据配置
    # ========================
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate, S: univariate')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features: s/t/h/d/b/w/m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--result_path', type=str, default='./results/')

    # ========================
    # 序列配置
    # ========================
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # ========================
    # 模型通用配置
    # ========================
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

    # ========================
    # 检索模型通用配置
    # ========================
    parser.add_argument('--top_k', type=int, default=5)

    # PatternSearch
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
    parser.add_argument('--train_chunk_size', type=int, default=1024)

    # TS2VecSearch
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    # RAGSearch (专用参数名前缀 rag_)
    parser.add_argument('--rag_d_model', type=int, default=32)
    parser.add_argument('--rag_n_heads', type=int, default=4)
    parser.add_argument('--rag_epochs', type=int, default=10)
    parser.add_argument('--rag_batch_size', type=int, default=128)
    parser.add_argument('--rag_lr', type=float, default=0.001)
    parser.add_argument('--rag_weight_decay', type=float, default=1e-4)

    # ========================
    # GPU 配置
    # ========================
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)

    # ========================
    # 兼容接口（检索模型需要但不使用）
    # ========================
    parser.add_argument('--patience', type=int, default=100)  # 深度学习模型用，检索模型忽略
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 兼容接口

    # ========================
    # Embedding 配置（兼容接口）
    # ========================
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    # ========================
    # 自动设置数据路径和频率
    # ========================
    data_freq_map = {
        'ETTh1': ('ETTh1.csv', 'h'),
        'ETTh2': ('ETTh2.csv', 'h'),
        'ETTm1': ('ETTm1.csv', 't'),
        'ETTm2': ('ETTm2.csv', 't'),
    }
    if args.data in data_freq_map:
        default_path, default_freq = data_freq_map[args.data]
        if args.data_path == 'ETTh1.csv' and args.data != 'ETTh1':
            args.data_path = default_path
        if args.freq == 'h' and default_freq != 'h':
            args.freq = default_freq

    setting = f'{args.model}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'

    exp = Exp_Long_Term_Forecast(args)

    print(f'>>>>>>>start experiment: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)

    print(f'>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)

    print('Done!')


if __name__ == '__main__':
    main()
