import argparse
import os
from forecast.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # basic config
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'fusion', 'cosine_match', 'plot'],
                        help='single: 单模型训练/测试; fusion: XGBoost 融合; '
                             'cosine_match: MSE匹配基线; plot: 融合对比图')
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')
    parser.add_argument('--model', type=str, default='DLinear',
                        choices=['DLinear', 'PatchTST'],
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
    parser.add_argument('--n_samples', type=int, default=3,
                        help='cosine_match 模式: 绘图样本数')
    parser.add_argument('--top_k', type=int, default=10,
                        help='cosine_match 模式: top-k 匹配展示')
    parser.add_argument('--do_plot', action='store_true', default=False,
                        help='cosine_match 模式: 是否绘图')

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
    parser.add_argument('--result_path', type=str, default='./forecast/results/')

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

    setting = f'{args.model}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'

    if args.mode == 'cosine_match':
        import sys
        sys.argv = [
            'cosine_match',
            '--root_path', args.root_path,
            '--seq_len', str(args.seq_len),
            '--pred_len', str(args.pred_len),
            '--flags', args.flags,
            '--n_samples', str(args.n_samples),
            '--top_k', str(args.top_k),
            '--output_dir', os.path.join(
                args.result_path,
                f'CosineMatch_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'),
        ]
        if args.do_plot:
            sys.argv.append('--plot')

        from forecast.models.CosineMatch import main as cosine_main
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

        from forecast.models.PlotFusion import main as plot_main
        plot_main()

    elif args.mode == 'fusion':
        from forecast.fusion.stacking import XGBStacking
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
