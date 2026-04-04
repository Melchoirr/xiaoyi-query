import os
import gc
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.metrics import metric


class Exp_Long_Term_Forecast(Exp_Basic):
    """统一实验类 - 支持7种检索模型"""

    def __init__(self, args):
        super().__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _collect_data(self, data_loader, device):
        """从 DataLoader 收集数据"""
        all_X = []
        all_Y = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            # 转为 numpy
            batch_x_np = batch_x.numpy()
            batch_y_np = batch_y.numpy()

            # Y 截断到 pred_len
            if batch_y_np.ndim == 3:
                batch_y_np = batch_y_np[:, -self.args.pred_len:, :]
            elif batch_y_np.ndim == 2:
                batch_y_np = batch_y_np[:, -self.args.pred_len:]

            all_X.append(batch_x_np)
            all_Y.append(batch_y_np)

        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)
        return X, Y

    def _numpy_predict(self, X, device):
        """使用 numpy 数组进行预测"""
        # 将 numpy 转为 torch tensor 再转回 numpy
        # DataLoader 已经是 batch 形式，我们分批处理
        batch_size = self.args.batch_size
        n_samples = X.shape[0]
        n_preds_list = []

        self.model.model.eval() if hasattr(self.model.model, 'eval') else None

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch_x = torch.from_numpy(X[i:end]).float().to(device)
                batch_pred = self.model.predict(batch_x)
                # 转为 numpy
                if hasattr(batch_pred, 'numpy'):
                    batch_pred = batch_pred.cpu().numpy()
                n_preds_list.append(batch_pred)

        return np.concatenate(n_preds_list, axis=0)

    def train(self, setting):
        """
        检索模型训练流程：
        1. 收集训练数据
        2. 构建记忆库（fit）
        3. 验证
        4. 测试
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        print("=" * 60)
        print(f"[{self.args.model}] Building memory bank from training data...")
        print("=" * 60)

        # 收集训练数据
        X_train, Y_train = self._collect_data(train_loader, self.device)
        print(f"[{self.args.model}] X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

        # 构建记忆库
        self.model.fit(X_train, Y_train)

        # 验证
        print(f"\nValidating...")
        X_val, Y_val = self._collect_data(vali_loader, self.device)
        Y_val_pred = self._numpy_predict(X_val, self.device)
        mae, mse, rmse, mape, mspe = metric(Y_val_pred, Y_val)
        print(f"Vali MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # 测试
        print(f"\nTesting...")
        self.test(setting, test=1)

        return self.model

    def test(self, setting, test=0, flag='test'):
        """测试模型"""
        data_set, data_loader = self._get_data(flag=flag)

        preds = []
        trues = []

        result_path = os.path.join(self.args.result_path, setting)
        if flag in ('val', 'train'):
            result_path = os.path.join(result_path, flag)
        os.makedirs(result_path, exist_ok=True)

        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            # 模型预测
            batch_pred = self.model.predict(batch_x.float())
            if hasattr(batch_pred, 'numpy'):
                batch_pred = batch_pred.cpu().numpy()

            # 真实值
            batch_y_np = batch_y.numpy()
            if batch_y_np.ndim == 3:
                batch_y_np = batch_y_np[:, -self.args.pred_len:, :]
            elif batch_y_np.ndim == 2:
                batch_y_np = batch_y_np[:, -self.args.pred_len:]

            preds.append(batch_pred)
            trues.append(batch_y_np)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print(f'{flag} shape: preds={preds.shape}, trues={trues.shape}')

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}')
        print(f'RESULT|{setting}|{flag}|mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}')

        np.save(os.path.join(result_path, 'pred.npy'), preds)
        np.save(os.path.join(result_path, 'true.npy'), trues)
        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))

        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe}
