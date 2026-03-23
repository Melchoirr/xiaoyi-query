"""
数据加载器模块
负责读取ETT CSV数据，进行标准化归一化，通过滑动窗口生成训练样本
严格遵循thuml/Time-Series-Library的学术规范结构
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_ETT_hour(Dataset):
    """
    ETT数据集加载器
    支持滑动窗口生成样本，严格的数据划分（Train/Val/Test = 7:1:2）
    """

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
        """
        Args:
            root_path: 数据根目录路径
            data_path: CSV文件名
            flag: 'train', 'val', 'test' 之一
            seq_len: 输入序列长度（历史窗口）
            pred_len: 预测序列长度（未来窗口）
            features: 'M' 多变量预测多变量, 'S' 单变量预测单变量
            target: 目标列名（当features='S'时使用）
            scale: 是否进行标准化
        """
        assert flag in ['train', 'val', 'test'], \
            f"flag必须为'train', 'val'或'test'，得到'{flag}'"
        assert features in ['M', 'S'], \
            f"features必须为'M'或'S'，得到'{features}'"

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale

        # 数据划分比例
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2

        # 初始化
        self.__read_data__()
        self.__split_data__()

    def __read_data__(self):
        """读取CSV数据并进行预处理"""
        # 完整数据路径
        self.data_path = os.path.join(self.root_path, self.data_path)

        # 读取数据，第一列为日期时间
        df_data = pd.read_csv(self.data_path)

        # 获取数值列（排除date列）
        cols_data = df_data.columns[1:]

        # 根据features模式选择数据
        if self.features == 'M':
            # 多变量模式：使用所有特征列
            self.raw_data = df_data[cols_data].values
        else:
            # 单变量模式：只使用目标列
            self.raw_data = df_data[[self.target]].values

        # 记录特征数量
        self.n_feature = self.raw_data.shape[1]

        # 标准化
        self.scaler = StandardScaler()
        if self.scale:
            self.raw_data = self.scaler.fit_transform(self.raw_data)

    def __split_data__(self):
        """按时间顺序划分数据集（不打乱）"""
        total_len = len(self.raw_data)
        train_len = int(total_len * self.train_ratio)
        val_len = int(total_len * self.val_ratio)

        # 划分边界
        border1s = [0, train_len, train_len + val_len]
        border2s = [train_len, train_len + val_len, total_len]

        # 根据flag选择划分
        if self.flag == 'test':
            self.border_start = border1s[2]
            self.border_end = border2s[2]
        elif self.flag == 'val':
            self.border_start = border1s[1]
            self.border_end = border2s[1]
        else:
            self.border_start = border1s[0]
            self.border_end = border2s[0]

        # 计算可用样本数（滑动窗口）
        self.data_len = self.border_end - self.border_start
        self.n_samples = self.data_len - self.seq_len - self.pred_len + 1

    def __len__(self):
        """返回数据集样本数量"""
        return max(0, self.n_samples)

    def __getitem__(self, index: int):
        """
        获取单个样本
        Returns:
            X: 输入序列 [seq_len, n_feature]
            Y: 目标序列 [pred_len, n_feature]
        """
        if index < 0 or index >= self.n_samples:
            raise IndexError(f"Index {index} out of range [0, {self.n_samples})")

        # 计算在完整数据中的起始位置
        start_idx = self.border_start + index
        end_idx = start_idx + self.seq_len + self.pred_len

        # 提取输入和目标
        X = self.raw_data[start_idx:start_idx + self.seq_len]
        Y = self.raw_data[start_idx + self.seq_len:end_idx]

        return X, Y

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反归一化，将标准化数据还原为原始尺度

        Args:
            data: 标准化后的数据，支持以下形状:
                - [n_samples, pred_len, n_features] (3D)
                - [n_samples, n_features] (2D)
                - [n_features] (1D)

        Returns:
            反归一化后的原始尺度数据，形状与输入一致
        """
        if not self.scale:
            return data

        original_shape = data.shape

        if data.ndim == 3:
            # 处理3D数据: [n, T, d] -> [n*T, d] -> 反归一化 -> [n, T, d]
            n, T, d = data.shape
            data_2d = data.reshape(-1, d)
            result_2d = self.scaler.inverse_transform(data_2d)
            result = result_2d.reshape(n, T, d)
        elif data.ndim == 2:
            # 处理2D数据
            result = self.scaler.inverse_transform(data)
        else:
            # 处理1D数据: [d] -> [1, d] -> 反归一化 -> [d]
            data_2d = data.reshape(1, -1)
            result_2d = self.scaler.inverse_transform(data_2d)
            result = result_2d.reshape(-1)

        return result


def get_data(args):
    """
    工厂函数：根据参数创建数据集

    Args:
        args: 包含以下属性的参数对象:
            - data_path: 数据文件名
            - root_path: 数据根目录
            - seq_len: 输入序列长度
            - pred_len: 预测序列长度
            - features: 'M' 或 'S'
            - target: 目标列名

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # 数据集划分
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
    从数据集对象中提取所有X和Y样本

    Args:
        dataset: Dataset_ETT_hour实例

    Returns:
        X_all: 所有输入样本 [n_samples, seq_len, n_feature]
        Y_all: 所有目标样本 [n_samples, pred_len, n_feature]
    """
    X_all = []
    Y_all = []

    for i in range(len(dataset)):
        X, Y = dataset[i]
        X_all.append(X)
        Y_all.append(Y)

    return np.array(X_all), np.array(Y_all)
