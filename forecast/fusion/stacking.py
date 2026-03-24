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
        self.xgb_model = None

    def _load_predictions(self, flag='test'):
        """加载所有模型的预测结果。

        Returns:
            stacked: [N, pred_len, C, num_models]
            true: [N, pred_len, C]
        """
        all_preds = []
        true = None

        for model_name in self.model_names:
            setting = self.setting_template.replace('{model}', model_name)
            if flag == 'val':
                pred_dir = os.path.join(self.result_path, setting, 'val')
            else:
                pred_dir = os.path.join(self.result_path, setting)

            pred_path = os.path.join(pred_dir, 'pred.npy')
            true_path = os.path.join(pred_dir, 'true.npy')

            if not os.path.exists(pred_path):
                raise FileNotFoundError(
                    f"未找到 {model_name} 的 {flag} 预测: {pred_path}\n"
                    f"请先运行: python -m forecast.run --model {model_name} ... --save_val_pred"
                )

            pred = np.load(pred_path)
            all_preds.append(pred)

            if true is None:
                true = np.load(true_path)

        # 检查所有模型预测样本数一致
        shapes = [p.shape for p in all_preds]
        if len(set(s[0] for s in shapes)) > 1:
            min_n = min(s[0] for s in shapes)
            print(f"警告: 模型预测样本数不一致 {shapes}，截断到 {min_n}")
            all_preds = [p[:min_n] for p in all_preds]
            true = true[:min_n]

        stacked = np.stack(all_preds, axis=-1)  # [N, pred_len, C, num_models]
        return stacked, true

    def _build_features(self, stacked):
        """构造 XGBoost 特征矩阵。

        Args:
            stacked: [N, pred_len, C, num_models]

        Returns:
            X: [N*pred_len*C, num_models + 1]  (各模型预测 + horizon_idx)
        """
        N, pred_len, C, M = stacked.shape
        X = stacked.reshape(-1, M)  # [N*pred_len*C, num_models]

        # horizon_idx: 每个预测步的位置，让 XGBoost 学到不同 horizon 的最优权重
        horizon = np.arange(pred_len)
        # 对每个样本的每个 horizon，重复 C 次（通道数）
        horizon_idx = np.tile(np.repeat(horizon, C), N)  # [N*pred_len*C]
        X = np.column_stack([X, horizon_idx])

        return X

    def train(self):
        """用 val 集上各模型的预测训练 XGBoost。"""
        stacked, true = self._load_predictions(flag='val')
        N, pred_len, C, M = stacked.shape
        print(f"融合训练: {len(self.model_names)} 个模型, "
              f"val 样本={N}, pred_len={pred_len}, channels={C}")

        X = self._build_features(stacked)
        y = true.reshape(-1)

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='reg:squarederror',
            n_jobs=-1,
        )
        self.xgb_model.fit(X, y)

        # 打印特征重要性
        importance = self.xgb_model.feature_importances_
        feature_names = self.model_names + ['horizon_idx']
        print("特征重要性:")
        for name, imp in zip(feature_names, importance):
            print(f"  {name}: {imp:.4f}")

    def predict_and_evaluate(self):
        """用 test 集推理并评估融合效果。

        Returns:
            (mae, mse, rmse, mape, mspe)
        """
        stacked, true = self._load_predictions(flag='test')
        N, pred_len, C, _ = stacked.shape

        X = self._build_features(stacked)
        y_pred = self.xgb_model.predict(X)
        y_pred = y_pred.reshape(N, pred_len, C)

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
        print(f"RESULT|{fusion_setting}|mse={mse:.6f}|mae={mae:.6f}|"
              f"rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}")
        print(f"结果已保存到: {fusion_path}")

        return mae, mse, rmse, mape, mspe
