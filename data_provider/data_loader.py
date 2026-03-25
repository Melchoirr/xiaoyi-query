"""
数据加载器模块 - 对齐 TSLib 学术规范
严格采用固定边界切分，支持时间特征编码，返回 4 值 (seq_x, seq_y, seq_x_mark, seq_y_mark)
内存优化：pd.read_csv 后立即 astype(float32) + del df_data + gc.collect()
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_ETT_hour(Dataset):
    """
    ETT 数据集加载器（TSLib 规范）

    关键改动：
    - 固定边界（非 ratio）：border1/2 按月份时间戳计算
    - __getitem__ 返回 4 值：seq_x, seq_y, seq_x_mark, seq_y_mark
    - 时间特征编码：Hour-of-Day + Day-of-Week
    - 全程 float32 + 早 del 释放
    """

    DTYPE = np.float32

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = 'train',
        seq_len: int = 96,
        pred_len: int = 48,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True
    ):
        assert flag in ['train', 'val', 'test']
        assert features in ['M', 'S']

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale

        # ── TSLib 固定边界（每小时 1 条，12 个月 = 12*30*24 = 8640 条/月）──
        self.border1s = []
        self.border2s = []

        self.__read_data__()
        self.__split_data__()

    def __read_data__(self):
        """读取 CSV -> 立即 float32 + del pandas 对象"""
        full_path = os.path.join(self.root_path, self.data_path)

        df_data = pd.read_csv(full_path)

        # ── 时间戳编码（Hour-of-Day + Day-of-Week）──
        # 假设第一列为 datetime 列（ETT 数据集格式）
        if 'date' in df_data.columns[0].lower() or df_data.columns[0] == df_data.columns[0]:
            ts_col = df_data.columns[0]
        else:
            ts_col = df_data.columns[0]

        try:
            timestamps = pd.to_datetime(df_data[ts_col])
        except Exception:
            timestamps = None

        if timestamps is not None:
            hour_of_day = timestamps.dt.hour.values.astype(self.DTYPE) / 23.0   # [0,1]
            day_of_week = timestamps.dt.dayofweek.values.astype(self.DTYPE) / 6.0    # [0,1]
            self.time_mark = np.stack([hour_of_day, day_of_week], axis=1)           # (N, 2)
        else:
            self.time_mark = np.zeros((len(df_data), 2), dtype=self.DTYPE)

        # 数值列 + 立即 float32
        cols_data = df_data.columns[1:]
        if self.features == 'M':
            raw = df_data[cols_data].values.astype(self.DTYPE)
        else:
            raw = df_data[[self.target]].values.astype(self.DTYPE)

        self.n_feature = raw.shape[1]

        # 标准化（StandardScaler fit 后立即降为 float32）
        self.scaler = StandardScaler()
        if self.scale:
            normalized = self.scaler.fit_transform(raw).astype(self.DTYPE)
        else:
            normalized = raw

        self.raw_data = normalized
        self._timestamp_len = len(self.raw_data)

        del raw, normalized, df_data
        gc.collect()

    def __split_data__(self):
        """TSLib 固定边界：按月份时间戳切分（hourly: 12*30*24=8640 条/月）"""
        total_len = self._timestamp_len
        month_len = 24 * 30 * 12          # 8640 per month (hourly data)

        # border1: 起始索引（不含 seq_len）
        set1 = 1 * month_len - self.seq_len   # 1 month - seq_len
        set2 = 4 * month_len - self.seq_len   # 4 months - seq_len
        set3 = 8 * month_len - self.seq_len   # 8 months - seq_len
        # 修正：TSLib 实际取 set3 = total_len（完整数据末尾）
        set3 = total_len

        self.border1s = [0, set1, set2]
        self.border2s = [set1, set2, set3]

        if self.flag == 'test':
            self.border_start = self.border1s[2]
            self.border_end = self.border2s[2]
        elif self.flag == 'val':
            self.border_start = self.border1s[1]
            self.border_end = self.border2s[1]
        else:
            self.border_start = self.border1s[0]
            self.border_end = self.border2s[0]

        self.data_len = self.border_end - self.border_start
        self.n_samples = max(0, self.data_len - self.seq_len - self.pred_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        """返回 4 值：seq_x, seq_y, seq_x_mark, seq_y_mark（均为 float32）"""
        if index < 0 or index >= self.n_samples:
            raise IndexError(f"Index {index} out of range [0, {self.n_samples})")

        start_idx = self.border_start + index
        end_idx = start_idx + self.seq_len + self.pred_len

        # 序列数据
        seq_x = self.raw_data[start_idx:start_idx + self.seq_len]           # (seq_len, n_feat)
        seq_y = self.raw_data[start_idx + self.seq_len:end_idx]             # (pred_len, n_feat)

        # 时间标记（按月份时间索引对应位置）
        seq_x_mark = self.time_mark[start_idx:start_idx + self.seq_len]     # (seq_len, 2)
        seq_y_mark = self.time_mark[start_idx + self.seq_len:end_idx]      # (pred_len, 2)

        return (
            seq_x.astype(self.DTYPE),
            seq_y.astype(self.DTYPE),
            seq_x_mark.astype(self.DTYPE),
            seq_y_mark.astype(self.DTYPE),
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反归一化，结果保持 float32"""
        if not self.scale:
            return data.astype(self.DTYPE)

        original_shape = data.shape

        if data.ndim == 3:
            n, T, d = data.shape
            result_2d = self.scaler.inverse_transform(data.reshape(-1, d))
            result = result_2d.reshape(n, T, d)
        elif data.ndim == 2:
            result = self.scaler.inverse_transform(data)
        else:
            data_2d = data.reshape(1, -1)
            result_2d = self.scaler.inverse_transform(data_2d)
            result = result_2d.reshape(-1)

        return result.astype(self.DTYPE)


def get_data(args):
    """
    工厂函数：根据参数创建数据集
    """
    train_set = Dataset_ETT_hour(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        scale=True
    )

    val_set = Dataset_ETT_hour(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='val',
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        scale=True
    )

    test_set = Dataset_ETT_hour(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='test',
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        scale=True
    )

    return train_set, val_set, test_set


def get_X_Y_from_dataset(dataset: Dataset_ETT_hour):
    """
    从数据集对象中提取 X 和 Y（忽略时间标记，适配基线模型接口）

    适配 __getitem__ 返回 4 值的解包。
    """
    n_samples = len(dataset)
    seq_len = dataset.seq_len
    n_feature = dataset.n_feature
    pred_len = dataset.pred_len

    X_all = np.empty((n_samples, seq_len, n_feature), dtype=Dataset_ETT_hour.DTYPE)
    Y_all = np.empty((n_samples, pred_len, n_feature), dtype=Dataset_ETT_hour.DTYPE)

    for i in range(n_samples):
        # 解包 4 值，只取前两个
        X, Y, _, _ = dataset[i]
        X_all[i] = X
        Y_all[i] = Y

    dataset.raw_data = None
    gc.collect()

    return X_all, Y_all


def get_X_Y_mark_from_dataset(dataset: Dataset_ETT_hour):
    """
    返回完整 4 元组（X, Y, X_mark, Y_mark）
    用于需要时间特征的模型。
    """
    n_samples = len(dataset)
    seq_len = dataset.seq_len
    n_feature = dataset.n_feature
    pred_len = dataset.pred_len
    mark_dim = dataset.time_mark.shape[1]   # 2 (hour + weekday)

    X_all  = np.empty((n_samples, seq_len, n_feature), dtype=Dataset_ETT_hour.DTYPE)
    Y_all  = np.empty((n_samples, pred_len, n_feature), dtype=Dataset_ETT_hour.DTYPE)
    Xm_all = np.empty((n_samples, seq_len, mark_dim),  dtype=Dataset_ETT_hour.DTYPE)
    Ym_all = np.empty((n_samples, pred_len, mark_dim),  dtype=Dataset_ETT_hour.DTYPE)

    for i in range(n_samples):
        X, Y, Xm, Ym = dataset[i]
        X_all[i]  = X
        Y_all[i]  = Y
        Xm_all[i] = Xm
        Ym_all[i] = Ym

    dataset.raw_data = None
    gc.collect()

    return X_all, Y_all, Xm_all, Ym_all
