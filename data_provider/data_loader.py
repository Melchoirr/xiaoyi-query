from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class _BaseETTDataset(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be train/val/test")

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path

        self.scaler = StandardScaler()
        self._read_data()

    def _get_borders(self):
        raise NotImplementedError

    def _read_data(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s, border2s = self._get_borders()
        type_map = {"train": 0, "val": 1, "test": 2}
        split_idx = type_map[self.flag]
        border1 = border1s[split_idx]
        border2 = border2s[split_idx]

        if self.features in {"M", "MS"}:
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be M/MS/S")

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].dt.month
            df_stamp["day"] = df_stamp["date"].dt.day
            df_stamp["weekday"] = df_stamp["date"].dt.weekday
            df_stamp["hour"] = df_stamp["date"].dt.hour
            if self.freq == "t":
                df_stamp["minute"] = df_stamp["date"].dt.minute
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)
        self.data_stamp = data_stamp.astype(np.float32)
        self.border1 = border1
        self.border2 = border2

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        arr = np.asarray(data)
        if arr.ndim == 2:
            return self.scaler.inverse_transform(arr)
        if arr.ndim == 3:
            b, l, c = arr.shape
            inv = self.scaler.inverse_transform(arr.reshape(-1, c))
            return inv.reshape(b, l, c)
        raise ValueError("data must be 2D or 3D")


class Dataset_ETT_hour(_BaseETTDataset):
    def _get_borders(self):
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        return border1s, border2s


class Dataset_ETT_minute(_BaseETTDataset):
    def _get_borders(self):
        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        return border1s, border2s


Dataset_Custom = Dataset_ETT_hour
