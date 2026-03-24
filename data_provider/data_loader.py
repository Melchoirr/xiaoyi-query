"""
数据加载器模块
负责读取ETT CSV数据，进行标准化归一化，通过滑动窗口生成训练样本
严格遵循thuml/Time-Series-Library的学术规范结构

内存优化策略：
- 所有 numpy 数组强制使用 float32（相比 float64 节省 50% 基础内存）
- 滑动窗口直接构造为目标 dtype，避免副本转换
- 大对象使用完毕后立即 del + gc.collect()
- 避免在内存中保留冗余的大表副本
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_ETT_hour(Dataset):
    """
    ETT数据集加载器
    支持滑动窗口生成样本，严格的数据划分（Train/Val/Test = 7:1:2）

    内存优化点：
    - 读取 CSV 时立即转换为 float32
    - __getitem__ 返回 float32，避免 dtype 不一致
    """

    # 全局 float32 dtype 常量
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

        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2

        # 初始化（按顺序执行，__read_data__ 会立即转换 dtype 并释放原始 df）
        self.__read_data__()
        self.__split_data__()

    def __read_data__(self):
        """读取CSV数据 -> float32 归一化 -> 立即释放中间对象"""
        full_path = os.path.join(self.root_path, self.data_path)

        # 1. 读取 CSV（pandas 读入后默认 float64）
        df_data = pd.read_csv(full_path)
        cols_data = df_data.columns[1:]

        # 2. 根据 features 模式选择列，直接构造成 float32，避免后续转换
        if self.features == 'M':
            raw = df_data[cols_data].values.astype(self.DTYPE)
        else:
            raw = df_data[[self.target]].values.astype(self.DTYPE)

        self.n_feature = raw.shape[1]

        # 3. 标准化（StandardScaler 内部使用 float64，过渡后立刻降为 float32）
        self.scaler = StandardScaler()
        if self.scale:
            normalized = self.scaler.fit_transform(raw).astype(self.DTYPE)
        else:
            normalized = raw

        # 4. 直接赋值给 self.raw_data，del 中间变量
        self.raw_data = normalized
        del raw, normalized, df_data
        gc.collect()

    def __split_data__(self):
        """按时间顺序划分数据集"""
        total_len = len(self.raw_data)
        train_len = int(total_len * self.train_ratio)
        val_len = int(total_len * self.val_ratio)

        border1s = [0, train_len, train_len + val_len]
        border2s = [train_len, train_len + val_len, total_len]

        if self.flag == 'test':
            self.border_start = border1s[2]
            self.border_end = border2s[2]
        elif self.flag == 'val':
            self.border_start = border1s[1]
            self.border_end = border2s[1]
        else:
            self.border_start = border1s[0]
            self.border_end = border2s[0]

        self.data_len = self.border_end - self.border_start
        self.n_samples = max(0, self.data_len - self.seq_len - self.pred_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        """返回 float32 样本，无副本"""
        if index < 0 or index >= self.n_samples:
            raise IndexError(f"Index {index} out of range [0, {self.n_samples})")

        start_idx = self.border_start + index
        end_idx = start_idx + self.seq_len + self.pred_len

        # 直接从 self.raw_data 切片，强制 float32（已经一致，无需 copy）
        X = self.raw_data[start_idx:start_idx + self.seq_len]
        Y = self.raw_data[start_idx + self.seq_len:end_idx]

        return X.astype(self.DTYPE), Y.astype(self.DTYPE)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反归一化，结果保持 float32"""
        if not self.scale:
            return data.astype(self.DTYPE)

        original_shape = data.shape

        if data.ndim == 3:
            n, T, d = data.shape
            data_2d = data.reshape(-1, d)
            result_2d = self.scaler.inverse_transform(data_2d)
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
    工厂函数：根据参数创建数据集（三个数据集各自独立，共享 scaler 归一化结果）

    优化：三个 Dataset 实例各自读 CSV，会产生三份 raw_data。
    对于大数据集，可在 get_X_Y_from_dataset 中逐步构建 X/Y 并立即释放 raw_data。
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
    从数据集对象中提取所有 X 和 Y 样本

    内存优化：
    - 使用 np.empty 预分配内存（比 append 列表再转 np.array 更高效）
    - 每行直接赋值，确保 float32 dtype
    - 提取完毕后手动 del 大对象，触发 gc
    """
    DTYPE = Dataset_ETT_hour.DTYPE
    n_samples = len(dataset)
    seq_len = dataset.seq_len
    n_feature = dataset.n_feature
    pred_len = dataset.pred_len

    # 预分配 float32 数组
    X_all = np.empty((n_samples, seq_len, n_feature), dtype=DTYPE)
    Y_all = np.empty((n_samples, pred_len, n_feature), dtype=DTYPE)

    for i in range(n_samples):
        X, Y = dataset[i]
        X_all[i] = X
        Y_all[i] = Y

    # 立即释放 dataset 中的 raw_data（可选，调用方若还需 dataset 可跳过）
    dataset.raw_data = None
    gc.collect()

    return X_all, Y_all
