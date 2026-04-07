import os
import numpy as np
import xgboost as xgb

from forecast.utils.metrics import metric


class XGBStacking:
    def __init__(self, model_names, result_path, setting_template):
        """
        model_names: ['DLinear', 'PatchTST', ...]
        result_path: './forecast/results/'
        setting_template: '{model}_ETTh1_M_sl96_pl96' ({model} 会被替换)
        """
        self.model_names = model_names
        self.result_path = result_path
        self.setting_template = setting_template
        self.xgb_models = {}  # channel_idx -> XGBRegressor

    def _load_predictions(self, flag='test'):
        """加载所有模型的预测结果。

        Args:
            flag: 'test', 'val', 'train', 或列表如 ['train', 'val'] 表示拼接多个集合

        Returns:
            stacked: [N, pred_len, C, num_models]
            true: [N, pred_len, C]
        """
        flags = flag if isinstance(flag, list) else [flag]

        all_preds_per_flag = []
        all_trues_per_flag = []

        for f in flags:
            preds_this_flag = []
            true_this_flag = None

            for model_name in self.model_names:
                setting = self.setting_template.replace('{model}', model_name)
                if f in ('val', 'train'):
                    pred_dir = os.path.join(self.result_path, setting, f)
                else:
                    pred_dir = os.path.join(self.result_path, setting)

                pred_path = os.path.join(pred_dir, 'pred.npy')
                true_path = os.path.join(pred_dir, 'true.npy')

                if not os.path.exists(pred_path):
                    raise FileNotFoundError(
                        f"未找到 {model_name} 的 {f} 预测: {pred_path}\n"
                        f"请先运行: python -m forecast.run --model {model_name} ... --save_val_pred --save_train_pred"
                    )

                pred = np.load(pred_path)
                preds_this_flag.append(pred)

                if true_this_flag is None:
                    true_this_flag = np.load(true_path)

            # 检查所有模型预测样本数一致
            shapes = [p.shape for p in preds_this_flag]
            if len(set(s[0] for s in shapes)) > 1:
                min_n = min(s[0] for s in shapes)
                print(f"警告: {f} 集模型预测样本数不一致 {shapes}，截断到 {min_n}")
                preds_this_flag = [p[:min_n] for p in preds_this_flag]
                true_this_flag = true_this_flag[:min_n]

            stacked = np.stack(preds_this_flag, axis=-1)  # [N, pred_len, C, num_models]
            all_preds_per_flag.append(stacked)
            all_trues_per_flag.append(true_this_flag)

        # 拼接所有 flag 的数据
        stacked = np.concatenate(all_preds_per_flag, axis=0)
        true = np.concatenate(all_trues_per_flag, axis=0)
        return stacked, true

    def train(self):
        """为每个通道单独训练一个 XGBoost。

        输入特征: 各模型在该通道的原始预测值 [N*pred_len, num_models]
        """
        stacked, true = self._load_predictions(flag=['train', 'val'])
        N, pred_len, C, M = stacked.shape
        print(f"融合训练: {len(self.model_names)} 个模型, "
              f"train+val 样本={N}, pred_len={pred_len}, channels={C}")

        for c in range(C):
            # stacked[:, :, c, :] -> [N, pred_len, M] -> [N*pred_len, M]
            X = stacked[:, :, c, :].reshape(-1, M)
            y = true[:, :, c].reshape(-1)

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='reg:squarederror',
                n_jobs=-1,
            )
            model.fit(X, y)
            self.xgb_models[c] = model

            # 打印特征重要性
            importance = model.feature_importances_
            imp_str = ', '.join(f'{m}={imp:.4f}' for m, imp in zip(self.model_names, importance))
            print(f"  通道 {c}: {imp_str}")

    def predict_and_evaluate(self):
        """用 test 集推理并评估融合效果。

        Returns:
            (mae, mse, rmse, mape, mspe)
        """
        stacked, true = self._load_predictions(flag='test')
        N, pred_len, C, _ = stacked.shape

        y_pred = np.zeros((N, pred_len, C))
        for c in range(C):
            X = stacked[:, :, c, :].reshape(-1, stacked.shape[-1])
            y_pred[:, :, c] = self.xgb_models[c].predict(X).reshape(N, pred_len)

        mae, mse, rmse, mape, mspe = metric(y_pred, true)

        # 保存融合结果
        fusion_setting = self.setting_template.replace('{model}', 'XGBFusion')
        fusion_path = os.path.join(self.result_path, fusion_setting)
        os.makedirs(fusion_path, exist_ok=True)

        np.save(os.path.join(fusion_path, 'pred.npy'), y_pred)
        np.save(os.path.join(fusion_path, 'true.npy'), true)
        np.save(os.path.join(fusion_path, 'metrics.npy'),
                np.array([mae, mse, rmse, mape, mspe]))

        print(f"融合结果: mse={mse:.4f}, mae={mae:.4f}")
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"RESULT|{ts}|{fusion_setting}|test|mse={mse:.6f}|mae={mae:.6f}|"
              f"rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}")
        print(f"结果已保存到: {fusion_path}")

        return mae, mse, rmse, mape, mspe
