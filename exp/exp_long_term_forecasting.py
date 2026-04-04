import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric


class Exp_Long_Term_Forecast(Exp_Basic):
    """统一实验类 - 支持检索模型和深度学习模型"""

    def __init__(self, args):
        super().__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        # 检索模型路径
        if getattr(self.model, 'is_retrieval', False):
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    outputs = self.model(batch_x)  # (B, pred_len, C)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
        else:
            # 深度学习模型路径
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 检索模型不需要梯度训练
        if getattr(self.model, 'is_retrieval', False):
            print("=" * 60)
            print(f"[Retrieval Model] No gradient training needed.")
            print(f"Building memory bank from training data...")
            print("=" * 60)

            # 构建记忆库
            all_X_train = []
            all_Y_train = []
            self.model.eval()

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    all_X_train.append(batch_x.cpu().numpy())
                    all_Y_train.append(batch_y.cpu().numpy())

            X_train = np.concatenate(all_X_train, axis=0)
            Y_train = np.concatenate(all_Y_train, axis=0)

            print(f"[Retrieval Model] X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

            # 调用模型的 fit 方法构建记忆库
            self.model.fit(X_train, Y_train)

            # 验证
            print(f"Validating...")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(f"Vali Loss: {vali_loss:.7f}")

            # 测试
            print(f"Testing...")
            self.test(setting, test=1)

            return self.model

        # 深度学习模型训练
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, flag='test'):
        data_set, _ = self._get_data(flag=flag)
        test_loader = DataLoader(data_set, batch_size=self.args.batch_size,
                                 shuffle=False, drop_last=False,
                                 num_workers=self.args.num_workers)

        if test:
            print('loading model')
            # 尝试加载 checkpoint（检索模型可能没有）
            model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))

        preds = []
        trues = []

        if flag in ('val', 'train'):
            result_path = os.path.join(self.args.result_path, setting, flag)
        else:
            result_path = os.path.join(self.args.result_path, setting)
        os.makedirs(result_path, exist_ok=True)

        self.model.eval()

        from tqdm import tqdm
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc=flag, ncols=80, ascii=True)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)
                outputs = outputs.detach().cpu()

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                preds.append(outputs.numpy())
                trues.append(batch_y.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print(f'{flag} shape: preds={preds.shape}, trues={trues.shape}')

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse:.4f}, mae:{mae:.4f}')
        print(f'RESULT|{setting}|{flag}|mse={mse:.6f}|mae={mae:.6f}|rmse={rmse:.6f}|mape={mape:.6f}|mspe={mspe:.6f}')

        np.save(os.path.join(result_path, 'pred.npy'), preds)
        np.save(os.path.join(result_path, 'true.npy'), trues)
        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))

        return
