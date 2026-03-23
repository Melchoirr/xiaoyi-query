"""
实验控制流模块
负责加载数据、初始化模型、执行测试、计算指标、保存结果
"""

import os
import numpy as np
from typing import Optional

from data_provider.data_loader import get_data, get_X_Y_from_dataset, Dataset_ETT_hour
from models.PatternSearch import PatternSearch
from utils.metrics import calculate_all_metrics, print_metrics, save_metrics_to_file


class Exp_Search:
    """
    PatternSearch 基线模型实验类
    封装完整的数据加载、模型训练/预测、评估流程
    """

    def __init__(self, args):
        """
        Args:
            args: 包含实验配置的参数对象
                - root_path: 数据根目录
                - data_path: 数据文件名
                - seq_len: 输入序列长度
                - pred_len: 预测序列长度
                - top_k: 近邻数量
                - features: 'M' 或 'S'
                - target: 目标列名
                - weighted: 是否使用逆距离加权
        """
        self.args = args
        self.model = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

        # 创建输出目录
        self.output_dir = './results'
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_model(self):
        """初始化 PatternSearch 模型"""
        self.model = PatternSearch(
            k=self.args.top_k,
            weighted=self.args.weighted,
            algorithm='kd_tree'
        )
        return self.model

    def _load_data(self):
        """加载并划分数据集"""
        print("\n" + "=" * 60)
        print("Loading and Preparing Data...")
        print("=" * 60)

        self.train_set, self.val_set, self.test_set = get_data(self.args)

        print(f"Data Path: {os.path.join(self.args.root_path, self.args.data_path)}")
        print(f"Features Mode: {self.args.features}")
        print(f"Sequence Length: {self.args.seq_len}")
        print(f"Prediction Length: {self.args.pred_len}")
        print(f"Train samples: {len(self.train_set)}")
        print(f"Val samples: {len(self.val_set)}")
        print(f"Test samples: {len(self.test_set)}")

        return self.train_set, self.val_set, self.test_set

    def _prepare_memory(self):
        """
        准备训练数据作为记忆库
        确保测试数据不泄露到记忆库中
        """
        print("\n" + "-" * 60)
        print("Preparing Memory Bank from Training Set...")
        print("-" * 60)

        X_train, Y_train = get_X_Y_from_dataset(self.train_set)
        print(f"Memory Bank X shape: {X_train.shape}")
        print(f"Memory Bank Y shape: {Y_train.shape}")

        return X_train, Y_train

    def _prepare_test_data(self):
        """准备测试数据"""
        X_test, Y_test = get_X_Y_from_dataset(self.test_set)
        print(f"Test Set X shape: {X_test.shape}")
        print(f"Test Set Y shape: {Y_test.shape}")
        return X_test, Y_test

    def fit(self):
        """
        构建记忆库（拟合模型）
        """
        # 加载数据
        self._load_data()

        # 构建模型
        self._build_model()

        # 准备记忆库数据
        X_train, Y_train = self._prepare_memory()

        # 在训练集上构建记忆索引
        print("\n" + "-" * 60)
        print(f"Fitting PatternSearch model (k={self.args.top_k}, weighted={self.args.weighted})...")
        print("-" * 60)
        self.model.fit(X_train, Y_train)
        print(f"Model Info: {self.model}")
        print("Memory index built successfully!")

        return self

    def predict(self) -> tuple:
        """
        对测试集进行预测

        Returns:
            preds: 预测结果 [n_test, pred_len, n_features]
            trues: 真实值 [n_test, pred_len, n_features]
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Please call fit() first.")

        print("\n" + "-" * 60)
        print("Running Prediction on Test Set...")
        print("-" * 60)

        # 准备测试数据
        X_test, Y_test = self._prepare_test_data()

        # 分批预测（避免内存问题）
        batch_size = 1000
        n_samples = X_test.shape[0]
        preds_list = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_X = X_test[i:end_idx]
            batch_preds = self.model.predict(batch_X, top_k=self.args.top_k)
            preds_list.append(batch_preds)

            if (i // batch_size + 1) % 5 == 0 or end_idx == n_samples:
                print(f"  Progress: {end_idx}/{n_samples} samples predicted")

        preds = np.concatenate(preds_list, axis=0)
        trues = Y_test

        print(f"Prediction completed! Shape: {preds.shape}")

        return preds, trues

    def test(self, save_results: bool = True) -> dict:
        """
        完整测试流程

        Args:
            save_results: 是否保存预测结果和指标

        Returns:
            metrics: 评估指标字典
        """
        # 拟合模型
        self.fit()

        # 预测
        preds, trues = self.predict()

        # 反归一化
        print("\n" + "-" * 60)
        print("Inverse Transforming Predictions and Ground Truth...")
        print("-" * 60)

        preds_original = self.test_set.inverse_transform(preds)
        trues_original = self.test_set.inverse_transform(trues)

        print(f"Original scale - Preds range: [{preds_original.min():.4f}, {preds_original.max():.4f}]")
        print(f"Original scale - Trues range: [{trues_original.min():.4f}, {trues_original.max():.4f}]")

        # 计算指标（在原始尺度上）
        print("\n" + "-" * 60)
        print("Calculating Evaluation Metrics (Original Scale)...")
        print("-" * 60)

        metrics = calculate_all_metrics(preds_original, trues_original)
        print_metrics(metrics)

        # 保存结果
        if save_results:
            self._save_results(preds_original, trues_original, metrics)

        return metrics

    def _save_results(self, preds: np.ndarray, trues: np.ndarray, metrics: dict):
        """
        保存预测结果和指标

        Args:
            preds: 预测值（原始尺度）
            trues: 真实值（原始尺度）
            metrics: 评估指标
        """
        # 生成文件名标识
        dataset_name = self.args.data_path.replace('.csv', '')
        exp_id = f"{dataset_name}_seq{self.args.seq_len}_pred{self.args.pred_len}_k{self.args.top_k}"

        # 保存预测结果为 .npy 文件
        preds_path = os.path.join(self.output_dir, f"{exp_id}_preds.npy")
        trues_path = os.path.join(self.output_dir, f"{exp_id}_trues.npy")

        np.save(preds_path, preds)
        np.save(trues_path, trues)
        print(f"\nPredictions saved to: {preds_path}")
        print(f"Ground truth saved to: {trues_path}")

        # 保存指标到 result.txt
        result_path = os.path.join(self.output_dir, "result.txt")
        print(f"Metrics saved to: {result_path}")

        # 写入详细信息
        with open(result_path, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Experiment: PatternSearch Baseline\n")
            f.write(f"Dataset: {self.args.data_path}\n")
            f.write(f"Features: {self.args.features}\n")
            f.write(f"seq_len: {self.args.seq_len}, pred_len: {self.args.pred_len}\n")
            f.write(f"top_k: {self.args.top_k}, weighted: {self.args.weighted}\n")
            f.write("-" * 60 + "\n")

            for key, value in metrics.items():
                if 'MAPE' in key or 'MSPE' in key:
                    f.write(f"{key}: {value:.6f}%\n")
                else:
                    f.write(f"{key}: {value:.6f}\n")

        print("\nResults saved successfully!")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        if self.model is None:
            return {}
        return self.model.get_params()


def run_experiment(args):
    """
    运行完整实验的便捷函数

    Args:
        args: 实验配置参数

    Returns:
        metrics: 评估指标字典
    """
    exp = Exp_Search(args)
    metrics = exp.test(save_results=True)
    return metrics
