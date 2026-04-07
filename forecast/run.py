import argparse
import os
from forecast.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # basic config
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'fusion', 'cosine_match', 'plot',
                                 'cross_var', 'cross_var_fusion', 'residual_fusion'],
                        help='single: 单模型训练/测试; fusion: XGBoost 融合; '
                             'cosine_match: MSE匹配基线; plot: 融合对比图; '
                             'cross_var: 跨变量预测(src_channel seq -> tgt_channel pred); '
                             'cross_var_fusion: 跨变量XGBoost融合')
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')
    parser.add_argument('--model', type=str, default='DLinear',
                        choices=['DLinear', 'PatchTST', 'PrimitiveFusion'],
                        help='single 模式下的模型选择')
    parser.add_argument('--fusion_models', type=str,
                        default='DLinear,PatchTST',
                        help='fusion 模式下参与融合的模型，逗号分隔')
    parser.add_argument('--save_val_pred', action='store_true', default=False,
                        help='测试后额外保存 val 集预测（供 fusion 使用）')
    parser.add_argument('--save_train_pred', action='store_true', default=False,
                        help='测试后额外保存 train 集预测（供 fusion 使用）')

    # cosine_match / plot 模式参数
    parser.add_argument('--flags', type=str, default='test,train,val',
                        help='cosine_match 模式: 逗号分隔的集合 (test,train,val)')
    parser.add_argument('--plot_models', type=str, default='DLinear,PatchTST',
                        help='plot 模式: 参与对比的基础模型名')
    parser.add_argument('--plot_fusion_model', type=str, default='XGBFusion',
                        help='plot 模式: 融合模型名')
    parser.add_argument('--plot_output', type=str, default=None,
                        help='plot 模式: 输出图片路径')
    parser.add_argument('--match_top_k', type=int, default=5,
                        help='cosine_match 模式: 匹配时取前K个最近邻加权平均')

    # cross_var 模式参数
    parser.add_argument('--src_channel', type=int, default=None,
                        help='cross_var: 输入变量的列索引 (0-based)')
    parser.add_argument('--tgt_channel', type=int, default=None,
                        help='cross_var: 预测目标变量的列索引 (0-based)')
    parser.add_argument('--fusion_train_flag', type=str, default='train_val',
                        choices=['train', 'train_val'],
                        help='cross_var_fusion: 用 train 还是 train+val 训练 XGBoost')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate, S: univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features: s/t/h/d/b/w/m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--result_path', type=str, default='./outputs/results/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # model config
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear individual channel')
    parser.add_argument('--revin', action='store_true', default=False,
                        help='enable Reversible Instance Normalization')

    # PatchTST config
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

    # PrimitiveFusion config
    parser.add_argument('--num_primitives', type=int, default=16,
                        help='number of primitive codebook entries (K)')
    parser.add_argument('--primitive_temp', type=float, default=1.0,
                        help='temperature for primitive soft-assignment')
    parser.add_argument('--n_cross_layers', type=int, default=1,
                        help='number of cross-attention layers for primitive fusion')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--num_workers', type=int, default=0)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)

    # embedding
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding: timeF, fixed, learned')

    args = parser.parse_args()

    # auto-set data_path and freq from data name
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

    if args.mode == 'cross_var':
        args.enc_in = 1
        args.features = 'M'  # 加载全部列，由 src/tgt_channel 切片
        setting = (f'{args.model}_{args.data}_crossvar'
                   f'_src{args.src_channel}_tgt{args.tgt_channel}'
                   f'_sl{args.seq_len}_pl{args.pred_len}')
    else:
        setting = f'{args.model}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'

    if args.mode == 'cosine_match':
        import sys
        sys.argv = [
            'cosine_match',
            '--root_path', args.root_path,
            '--seq_len', str(args.seq_len),
            '--pred_len', str(args.pred_len),
            '--flags', args.flags,
            '--match_top_k', str(args.match_top_k),
            '--output_dir', os.path.join(
                args.result_path,
                f'CosineMatch_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'),
        ]

        from forecast.baselines.CosineMatch import main as cosine_main
        cosine_main()

    elif args.mode == 'plot':
        import sys
        sys.argv = [
            'plot_fusion',
            '--seq_len', str(args.seq_len),
            '--pred_len', str(args.pred_len),
            '--data', args.data,
            '--features', args.features,
            '--models', args.plot_models,
            '--fusion_model', args.plot_fusion_model,
            '--result_path', args.result_path,
            '--raw_path', os.path.join(args.root_path, f'{args.data}.csv'),
        ]
        if args.plot_output:
            sys.argv += ['--output', args.plot_output]

        from scripts.plotting.plot_fusion import main as plot_main
        plot_main()

    elif args.mode in ('cross_var_fusion', 'residual_fusion'):
        setting_template = (f'DLinear_{args.data}_crossvar'
                            f'_src{{src}}_tgt{{tgt}}'
                            f'_sl{args.seq_len}_pl{args.pred_len}')
        flag_suffix = 'train' if args.fusion_train_flag == 'train' else 'trainval'

        if args.mode == 'cross_var_fusion':
            from forecast.fusion.cross_var_stacking import CrossVarStacking
            stacker = CrossVarStacking(args.enc_in, args.result_path, setting_template)
            stacker.train(flag=args.fusion_train_flag)
            stacker.predict_and_evaluate(fusion_name=f'CrossVarFusion_{flag_suffix}')
        else:
            from forecast.fusion.cross_var_stacking import ResidualStacking
            stacker = ResidualStacking(args.enc_in, args.result_path, setting_template)
            stacker.train(flag=args.fusion_train_flag)
            stacker.predict_and_evaluate(fusion_name=f'ResidualFusion_{flag_suffix}')

    elif args.mode == 'fusion':
        from forecast.fusion.stacking import XGBStacking
        setting_template = f'{{model}}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'
        model_names = [m.strip() for m in args.fusion_models.split(',')]
        stacker = XGBStacking(model_names, args.result_path, setting_template)
        stacker.train()
        stacker.predict_and_evaluate()
    elif args.mode in ('single', 'cross_var'):
        exp = Exp_Long_Term_Forecast(args)

        if args.is_training:
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
        else:
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)

        if args.save_val_pred:
            print(f'>>>>>>>saving val predictions : {setting}<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1, flag='val')

        if args.save_train_pred:
            print(f'>>>>>>>saving train predictions : {setting}<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1, flag='train')

    print('Done!')


if __name__ == '__main__':
    main()
