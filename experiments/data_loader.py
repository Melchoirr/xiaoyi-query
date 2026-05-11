from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from forecast.utils.timefeatures import time_features


FLAGS = ("train", "val", "test")
EPS = 1e-6


def _infer_data_name(data_path: str) -> str:
    name = os.path.splitext(os.path.basename(data_path))[0]
    if name in data_dict:
        return name
    return "custom"


def _timeenc_from_args(args) -> int:
    if hasattr(args, "timeenc"):
        return int(args.timeenc)
    return 0 if getattr(args, "embed", "timeF") != "timeF" else 1


def _default_freq(data_path: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    name = os.path.splitext(os.path.basename(data_path))[0].lower()
    return "t" if name.startswith("ettm") else "h"


def _select_columns(df_raw: pd.DataFrame, features: str, target: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    value_cols = list(df_raw.columns[1:])
    if target not in value_cols:
        raise ValueError(f"target column {target!r} not found in data")

    if features in ("M", "MS"):
        cols_data = value_cols
    elif features == "S":
        cols_data = [target]
    else:
        raise ValueError(f"unsupported features mode: {features}")
    return df_raw[cols_data], cols_data, cols_data


def _calendar_features(df_stamp: pd.DataFrame, freq: str) -> np.ndarray:
    df_stamp = df_stamp.copy()
    df_stamp["date"] = pd.to_datetime(df_stamp.date)
    df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
    if freq in ("t", "min", "15min"):
        df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
    return df_stamp.drop(["date"], axis=1).values.astype(np.float32)


@dataclass(frozen=True)
class SplitBorders:
    border1s: list[int]
    border2s: list[int]


class _BaseForecastDataset(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "val", "test"]
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _borders(self, df_raw: pd.DataFrame) -> SplitBorders:
        raise NotImplementedError

    def _load_raw(self) -> pd.DataFrame:
        path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset file not found: {path}")
        df_raw = pd.read_csv(path)
        if "date" not in df_raw.columns:
            raise ValueError(f"{path} must contain a 'date' column")
        return df_raw

    def _read_data(self) -> None:
        self.scaler = StandardScaler()
        df_raw = self._load_raw()
        borders = self._borders(df_raw)
        border1 = borders.border1s[self.set_type]
        border2 = borders.border2s[self.set_type]

        df_data, key_cols, pred_cols = _select_columns(df_raw, self.features, self.target)
        self.key_cols = key_cols
        self.pred_cols = pred_cols
        self.value_cols = list(df_data.columns)

        if self.scale:
            train_data = df_data[borders.border1s[0]:borders.border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[["date"]][border1:border2]
        if self.timeenc == 0:
            data_stamp = _calendar_features(df_stamp, self.freq)
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0).astype(np.float32)

        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)
        self.data_stamp = data_stamp

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
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(_BaseForecastDataset):
    def _borders(self, df_raw: pd.DataFrame) -> SplitBorders:
        train = 12 * 30 * 24
        val = 4 * 30 * 24
        test = 4 * 30 * 24
        return SplitBorders(
            border1s=[0, train - self.seq_len, train + val - self.seq_len],
            border2s=[train, train + val, train + val + test],
        )


class Dataset_ETT_minute(_BaseForecastDataset):
    def _borders(self, df_raw: pd.DataFrame) -> SplitBorders:
        unit = 4
        train = 12 * 30 * 24 * unit
        val = 4 * 30 * 24 * unit
        test = 4 * 30 * 24 * unit
        return SplitBorders(
            border1s=[0, train - self.seq_len, train + val - self.seq_len],
            border2s=[train, train + val, train + val + test],
        )


class Dataset_Custom(_BaseForecastDataset):
    def _load_raw(self) -> pd.DataFrame:
        df_raw = super()._load_raw()
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        return df_raw[["date"] + cols + [self.target]]

    def _borders(self, df_raw: pd.DataFrame) -> SplitBorders:
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test
        return SplitBorders(
            border1s=[0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            border2s=[num_train, num_train + num_val, len(df_raw)],
        )


data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
}


def data_provider(args, flag):
    data_name = getattr(args, "data", None) or _infer_data_name(args.data_path)
    if data_name not in data_dict:
        raise ValueError(f"unsupported data={data_name!r}; expected one of {sorted(data_dict)}")

    Data = data_dict[data_name]
    label_len = getattr(args, "label_len", None)
    if label_len is None:
        label_len = 48
    batch_size = getattr(args, "batch_size", 32)
    num_workers = getattr(args, "num_workers", 0)
    timeenc = _timeenc_from_args(args)
    freq = _default_freq(args.data_path, getattr(args, "freq", None))

    shuffle_flag = False if flag == "test" else getattr(args, "shuffle", False)
    drop_last = getattr(args, "drop_last", False)

    data_set = Data(
        args=args,
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=getattr(args, "seasonal_patterns", None),
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader


def collect_windows(data_loader, pred_len: int):
    history_parts = []
    future_parts = []
    for seq_x, seq_y, _, _ in data_loader:
        history_parts.append(np.asarray(seq_x, dtype=np.float32))
        future_parts.append(np.asarray(seq_y[:, -pred_len:, :], dtype=np.float32))
    return np.concatenate(history_parts, axis=0), np.concatenate(future_parts, axis=0)


def build_shape_features(windows: np.ndarray) -> np.ndarray:
    mean = windows.mean(axis=1, keepdims=True)
    std = np.maximum(windows.std(axis=1, keepdims=True), EPS)
    shape = (windows - mean) / std
    return shape.reshape(shape.shape[0], -1).astype(np.float32, copy=False)
