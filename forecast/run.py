import argparse
import torch
from forecast.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # basic config
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'fusion'],
                        help='single: 单模型训练/测试; fusion: XGBoost 融合')
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')
    parser.add_argument('--model', type=str, default='DLinear',
                        choices=['DLinear', 'PatchTST', 'Sundial', 'Chronos', 'Timer', 'Moirai'])
    parser.add_argument('--fusion_models', type=str,
                        default='DLinear,PatchTST',
                        help='fusion 模式下参与融合的模型，逗号分隔')
    parser.add_argument('--save_val_pred', action='store_true', default=False,
                        help='测试后额外保存 val 集预测（供 fusion 使用）')
    parser.add_argument('--save_train_pred', action='store_true', default=False,
                        help='测试后额外保存 train 集预测（供 fusion 使用）')

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

    # Foundation model configs
    parser.add_argument('--sundial_model', type=str, default='thuml/sundial-base-128m')
    parser.add_argument('--chronos_model', type=str, default='amazon/chronos-2')
    parser.add_argument('--timer_model', type=str, default='thuml/timer-base-84m')
    parser.add_argument('--moirai_model', type=str, default='Salesforce/moirai-1.1-R-base')

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

    # Foundation model device config
    ZERO_SHOT_MODELS = ('Sundial', 'Chronos', 'Timer', 'Moirai')
    if args.model in ZERO_SHOT_MODELS:
        if args.use_gpu and torch.cuda.is_available():
            args.device = f'cuda:{args.gpu}'
        elif args.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    setting = f'{args.model}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'

    if args.mode == 'fusion':
        from forecast.fusion.stacking import XGBStacking
        setting_template = f'{{model}}_{args.data}_{args.features}_sl{args.seq_len}_pl{args.pred_len}'
        model_names = [m.strip() for m in args.fusion_models.split(',')]
        stacker = XGBStacking(model_names, args.result_path, setting_template)
        stacker.train()
        stacker.predict_and_evaluate()
    else:
        exp = Exp_Long_Term_Forecast(args)

        if args.is_training and args.model not in ZERO_SHOT_MODELS:
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
        else:
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

        if args.save_val_pred:
            print(f'>>>>>>>saving val predictions : {setting}<<<<<<<<<<<<<<<<<<')
            load_ckpt = 0 if args.model in ZERO_SHOT_MODELS else 1
            exp.test(setting, test=load_ckpt, flag='val')

        if args.save_train_pred:
            print(f'>>>>>>>saving train predictions : {setting}<<<<<<<<<<<<<<<<<<')
            load_ckpt = 0 if args.model in ZERO_SHOT_MODELS else 1
            exp.test(setting, test=load_ckpt, flag='train')

    print('Done!')


if __name__ == '__main__':
    main()
