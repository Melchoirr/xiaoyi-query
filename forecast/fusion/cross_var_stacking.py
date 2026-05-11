import os
import numpy as np
import xgboost as xgb

from forecast.utils.metrics import metric


class CrossVarStacking:
    """跨变量 XGBoost 融合。

    对每个目标变量 tgt，将 7 个 source channel 的 DLinear 预测作为特征，
    训练 XGBoost 学习最优组合。
    """

    def __init__(self, n_channels, result_path, setting_template):
        """
        n_channels: 变量数（ETTh1=7）
        result_path: './outputs/results/'
        setting_template: 'DLinear_ETTh1_crossvar_src{src}_tgt{tgt}_sl192_pl96'
        """
        self.n_channels = n_channels
        self.result_path = result_path
        self.setting_template = setting_template
        self.xgb_models = {}  # tgt_channel -> XGBRegressor
        self.col_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

    def _load_cross_var_predictions(self, tgt, flag='test'):
        """加载某个 target 的所有 source 预测。

        Returns:
            stacked: [N, pred_len, n_channels]  (7 个 source 的预测)
            true: [N, pred_len, 1]
        """
        flags = flag if isinstance(flag, list) else [flag]

        all_stacked = []
        all_true = []

        for f in flags:
            preds = []
            true_data = None

            for src in range(self.n_channels):
                setting = self.setting_template.format(src=src, tgt=tgt)
                if f in ('val', 'train'):
                    pred_dir = os.path.join(self.result_path, setting, f)
                else:
                    pred_dir = os.path.join(self.result_path, setting)

                pred_path = os.path.join(pred_dir, 'pred.npy')
                true_path = os.path.join(pred_dir, 'true.npy')

                if not os.path.exists(pred_path):
                    raise FileNotFoundError(
                        f"未找到 src={src}->tgt={tgt} 的 {f} 预测: {pred_path}\n"
                        f"请先运行 cross_var 实验并加 --save_val_pred --save_train_pred"
                    )

                pred = np.load(pred_path)  # [N, pred_len, 1]
                preds.append(pred[:, :, 0])  # [N, pred_len]

                if true_data is None:
                    true_data = np.load(true_path)  # [N, pred_len, 1]

            # [N, pred_len, 7]
            stacked = np.stack(preds, axis=-1)
            all_stacked.append(stacked)
            all_true.append(true_data)

        stacked = np.concatenate(all_stacked, axis=0)
        true = np.concatenate(all_true, axis=0)
        return stacked, true

    def train(self, flag='train_val'):
        """训练 XGBoost 融合模型。

        Args:
            flag: 'train' 仅用 train 集, 'train_val' 用 train+val 集
        """
        train_flag = ['train', 'val'] if flag == 'train_val' else ['train']
        flag_desc = 'train+val' if flag == 'train_val' else 'train'

        print(f"\n跨变量融合训练 (XGBoost on {flag_desc} set)")
        print(f"{'='*60}")

        for tgt in range(self.n_channels):
            stacked, true = self._load_cross_var_predictions(tgt, flag=train_flag)
            N, pred_len, M = stacked.shape

            X = stacked.reshape(-1, M)          # [N*pred_len, 7]
            y = true[:, :, 0].reshape(-1)       # [N*pred_len]

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='reg:squarederror',
                n_jobs=-1,
            )
            model.fit(X, y)
            self.xgb_models[tgt] = model

            importance = model.feature_importances_
            imp_str = ', '.join(
                f'{self.col_names[s]}={imp:.4f}'
                for s, imp in enumerate(importance)
            )
            print(f"  tgt={self.col_names[tgt]}: samples={N}, "
                  f"importance: {imp_str}")

    def predict_and_evaluate(self, fusion_name='CrossVarFusion'):
        """在 test 集推理并评估。"""
        print(f"\n跨变量融合测试")
        print(f"{'='*60}")

        N = None
        pred_len = None
        all_preds = []
        all_trues = []

        for tgt in range(self.n_channels):
            stacked, true = self._load_cross_var_predictions(tgt, flag='test')
            if N is None:
                N, pred_len, _ = stacked.shape

            X = stacked.reshape(-1, self.n_channels)
            pred = self.xgb_models[tgt].predict(X).reshape(N, pred_len)
            all_preds.append(pred)
            all_trues.append(true[:, :, 0])

        # [N, pred_len, 7]
        y_pred = np.stack(all_preds, axis=-1)
        y_true = np.stack(all_trues, axis=-1)

        mae, mse, rmse, mape, mspe = metric(y_pred, y_true)

        # 逐通道指标
        print(f"\n  逐通道 MSE:")
        for c in range(self.n_channels):
            c_mae, c_mse, c_rmse, _, _ = metric(y_pred[:, :, c:c+1], y_true[:, :, c:c+1])
            print(f"    {self.col_names[c]}: MSE={c_mse:.4f}, MAE={c_mae:.4f}")

        # 保存
        fusion_setting = f'{fusion_name}_{self.setting_template.split("_")[1]}_crossvar' \
                         f'_sl{self.setting_template.split("sl")[1]}'
        # 简化 setting 名
        parts = self.setting_template.format(src=0, tgt=0).split('_')
        data_name = parts[1]
        sl_pl = '_'.join(p for p in parts if p.startswith('sl') or p.startswith('pl'))
        fusion_setting = f'{fusion_name}_{data_name}_crossvar_{sl_pl}'

        fusion_path = os.path.join(self.result_path, fusion_setting)
        os.makedirs(fusion_path, exist_ok=True)

        np.save(os.path.join(fusion_path, 'pred.npy'), y_pred)
        np.save(os.path.join(fusion_path, 'true.npy'), y_true)
        np.save(os.path.join(fusion_path, 'metrics.npy'),
                np.array([mae, mse, rmse, mape, mspe]))

        print(f"\n  整体: mse={mse:.4f}, mae={mae:.4f}")
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"RESULT|{ts}|{fusion_setting}|test|mse={mse:.6f}|mae={mae:.6f}|"
              f"rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}")
        print(f"结果已保存到: {fusion_path}")

        return mae, mse, rmse, mape, mspe


class ResidualStacking:
    """残差融合：DLinear 自回归 + XGBoost 拟合残差的非线性部分。

    对每个通道：
    1. DLinear 自回归预测 (src=tgt) 作为基线
    2. XGBoost 用其他 6 个跨变量预测作为特征，拟合自回归的残差
    3. 最终输出 = self_pred + xgb_residual
    """

    def __init__(self, n_channels, result_path, setting_template):
        self.n_channels = n_channels
        self.result_path = result_path
        self.setting_template = setting_template
        self.xgb_models = {}
        self.col_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

    def _load_pred(self, src, tgt, flag):
        """加载单个 (src, tgt) 对的预测"""
        setting = self.setting_template.format(src=src, tgt=tgt)
        if flag in ('val', 'train'):
            pred_dir = os.path.join(self.result_path, setting, flag)
        else:
            pred_dir = os.path.join(self.result_path, setting)
        pred = np.load(os.path.join(pred_dir, 'pred.npy'))[:, :, 0]  # [N, pred_len]
        true = np.load(os.path.join(pred_dir, 'true.npy'))[:, :, 0]  # [N, pred_len]
        return pred, true

    def _load_for_channel(self, tgt, flags):
        """加载某个 target 通道的自回归预测 + 跨变量特征"""
        all_self_pred, all_cross_feat, all_true = [], [], []

        for f in flags:
            self_pred, true = self._load_pred(tgt, tgt, f)  # 自回归

            cross_preds = []
            for src in range(self.n_channels):
                if src == tgt:
                    continue
                pred, _ = self._load_pred(src, tgt, f)
                cross_preds.append(pred)

            # cross features: [N, pred_len, 6]
            cross_feat = np.stack(cross_preds, axis=-1)
            all_self_pred.append(self_pred)
            all_cross_feat.append(cross_feat)
            all_true.append(true)

        self_pred = np.concatenate(all_self_pred, axis=0)
        cross_feat = np.concatenate(all_cross_feat, axis=0)
        true = np.concatenate(all_true, axis=0)
        return self_pred, cross_feat, true

    def train(self, flag='train_val'):
        train_flags = ['train', 'val'] if flag == 'train_val' else ['train']
        flag_desc = 'train+val' if flag == 'train_val' else 'train'

        print(f"\n残差融合训练 (DLinear self + XGBoost residual, {flag_desc})")
        print(f"{'='*60}")

        for tgt in range(self.n_channels):
            self_pred, cross_feat, true = self._load_for_channel(tgt, train_flags)
            N, pred_len = self_pred.shape

            residual = true - self_pred  # [N, pred_len]

            X = cross_feat.reshape(-1, self.n_channels - 1)  # [N*pred_len, 6]
            y = residual.reshape(-1)                          # [N*pred_len]

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='reg:squarederror',
                n_jobs=-1,
            )
            model.fit(X, y)
            self.xgb_models[tgt] = model

            # 特征名：除自己外的 6 个 source
            feat_names = [self.col_names[s] for s in range(self.n_channels) if s != tgt]
            importance = model.feature_importances_
            imp_str = ', '.join(f'{n}={imp:.4f}' for n, imp in zip(feat_names, importance))
            residual_std = np.std(residual)
            print(f"  tgt={self.col_names[tgt]}: residual_std={residual_std:.4f}, "
                  f"importance: {imp_str}")

    def predict_and_evaluate(self, fusion_name='ResidualFusion'):
        print(f"\n残差融合测试")
        print(f"{'='*60}")

        all_preds = []
        all_trues = []

        for tgt in range(self.n_channels):
            self_pred, cross_feat, true = self._load_for_channel(tgt, ['test'])
            N, pred_len = self_pred.shape

            X = cross_feat.reshape(-1, self.n_channels - 1)
            residual_pred = self.xgb_models[tgt].predict(X).reshape(N, pred_len)

            final_pred = self_pred + residual_pred
            all_preds.append(final_pred)
            all_trues.append(true)

        y_pred = np.stack(all_preds, axis=-1)  # [N, pred_len, 7]
        y_true = np.stack(all_trues, axis=-1)

        mae, mse, rmse, mape, mspe = metric(y_pred, y_true)

        print(f"\n  逐通道 MSE (残差融合 vs 自回归):")
        for c in range(self.n_channels):
            c_mae, c_mse, _, _, _ = metric(y_pred[:, :, c:c+1], y_true[:, :, c:c+1])
            # 自回归基线
            self_setting = self.setting_template.format(src=c, tgt=c)
            self_metrics = np.load(
                os.path.join(self.result_path, self_setting, 'metrics.npy'))
            self_mse = self_metrics[1]
            diff = c_mse - self_mse
            symbol = '↑' if diff > 0 else '↓'
            print(f"    {self.col_names[c]}: fusion={c_mse:.4f}, "
                  f"self={self_mse:.4f}, diff={diff:+.4f} {symbol}")

        # 保存
        parts = self.setting_template.format(src=0, tgt=0).split('_')
        data_name = parts[1]
        sl_pl = '_'.join(p for p in parts if p.startswith('sl') or p.startswith('pl'))
        fusion_setting = f'{fusion_name}_{data_name}_crossvar_{sl_pl}'

        fusion_path = os.path.join(self.result_path, fusion_setting)
        os.makedirs(fusion_path, exist_ok=True)

        np.save(os.path.join(fusion_path, 'pred.npy'), y_pred)
        np.save(os.path.join(fusion_path, 'true.npy'), y_true)
        np.save(os.path.join(fusion_path, 'metrics.npy'),
                np.array([mae, mse, rmse, mape, mspe]))

        print(f"\n  整体: mse={mse:.4f}, mae={mae:.4f}")
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"RESULT|{ts}|{fusion_setting}|test|mse={mse:.6f}|mae={mae:.6f}|"
              f"rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}")
        print(f"结果已保存到: {fusion_path}")

        return mae, mse, rmse, mape, mspe
