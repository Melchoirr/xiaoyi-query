"""
run.py - 统一时序预测框架 (支持检索模型 + 深度学习模型)

用法示例:
    # 检索模型
    python run.py --model PatternSearch --data ETTh1 --seq_len 512 --pred_len 96

    # 深度学习模型
    python run.py --model DLinear --data ETTh1 --seq_len 512 --pred_len 96 --is_training 1

    # 模型融合
    python run.py --mode fusion --fusion_models PatternSearch,DLinear
"""

import argparse
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting (Unified Framework)')

    # ========================
    # 基本配置
    # ========================
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'fusion'],
                        help='single: 单模型训练/测试; fusion: XGBoost 融合')
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')

    # ========================
    # 模型选择
    # ========================
    RETRIEVAL_CHOICES = [
        'PatternSearch', 'LSHSearch', 'SAXSearch', 'DTWSearch',
        'MatrixProfileSearch', 'TS2VecSearch', 'RAGSearch'
    ]
    DL_CHOICES = ['DLinear', 'PatchTST']
    ALL_MODEL_CHOICES = RETRIEVAL_CHOICES + DL_CHOICES

    parser.add_argument('--model', type=str, default='PatternSearch',
                        choices=ALL_MODEL_CHOICES,
                        help='模型选择')
    parser.add_argument('--fusion_models', type=str,
                        default='PatternSearch,DLinear',
                        help='fusion 模式下参与融合的模型，逗号分隔')

    # ========================
    # 数据配置
    # ========================
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate, S: univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features: s/t/h/d/b/w/m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--save_val_pred', action='store_true', default=False,
                        help='测试后额外保存 val 集预测（供 fusion 使用）')
    parser.add_argument('--save_train_pred', action='store_true', default=False,
                        help='测试后额外保存 train 集预测（供 fusion 使用）')

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
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear individual channel')
    parser.add_argument('--revin', action='store_true', default=False,
                        help='enable Reversible Instance Normalization')

    # ========================
    # 深度学习模型配置 (DLinear / PatchTST)
    # ========================
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fc_dropout', type=float, default=None,
                        help='fully-connected dropout (default: same as --dropout)')
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help='prediction head dropout')

    # ========================
    # 检索模型配置
    # ========================
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
    # 优化配置 (用于深度学习模型)
    # ========================
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--num_workers', type=int, default=0)

    # ========================
    # GPU 配置
    # ========================
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)

    # ========================
    # Embedding 配置
    # ========================
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding: timeF, fixed, learned')

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

    # fc_dropout defaults to dropout if not specified
    if args.fc_dropout is None:
        args.fc_dropout = args.dropout

    # ========================
    # 参数同步：RAGSearch 参数映射
    # ========================
    # 将 rag_ 前缀的参数同步到主参数
    args.d_model = args.rag_d_model      # 用于 RAGSearch
    args.n_heads = args.rag_n_heads      # 用于 RAGSearch
    args.epochs = args.rag_epochs        # 用于 RAGSearch
    args.batch_size = args.rag_batch_size  # 用于 RAGSearch
    args.lr = args.rag_lr                # 用于 RAGSearch
    args.weight_decay = args.rag_weight_decay  # 用于 RAGSearch

    setting = f'{args.model}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'

    if args.mode == 'fusion':
        from fusion.stacking import XGBStacking
        setting_template = f'{{model}}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'
        model_names = [m.strip() for m in args.fusion_models.split(',')]
        stacker = XGBStacking(model_names, args.result_path, setting_template)
        stacker.train()
        stacker.predict_and_evaluate()
    else:
        exp = Exp_Long_Term_Forecast(args)

        if args.is_training:
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
        else:
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

        if args.save_val_pred:
            print(f'>>>>>>>saving val predictions : {setting}<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1, flag='val')

        if args.save_train_pred:
            print(f'>>>>>>>saving train predictions : {setting}<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1, flag='train')

    print('Done!')


if __name__ == '__main__':
    main()
