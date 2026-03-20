"""
ONNX向量化引擎 - 使用PyTorch构建1D-CNN并导出为ONNX模型
"""

from typing import List
import numpy as np
import torch
import torch.nn as nn
import onnxruntime


class CNN1DEncoder(nn.Module):
    """简单的1D-CNN编码器模型"""

    def __init__(self, input_length: int = 100, embedding_dim: int = 128):
        super().__init__()
        self.input_length = input_length
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, input_length)

        Returns:
            128维嵌入向量，形状为 (batch_size, embedding_dim)
        """
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


class ONNXEncoder:
    """ONNX向量化编码器"""

    def __init__(self, onnx_path: str = "encoder_v1.onnx", input_length: int = 100):
        """
        初始化编码器

        Args:
            onnx_path: ONNX模型保存路径
            input_length: 输入序列长度
        """
        self.onnx_path = onnx_path
        self.input_length = input_length
        self.embedding_dim = 128

        self._build_and_export_model()

        self.session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _build_and_export_model(self) -> None:
        """使用PyTorch构建模型并导出为ONNX格式（仅当文件不存在时）"""
        import os
        if os.path.exists(self.onnx_path):
            return

        model = CNN1DEncoder(
            input_length=self.input_length,
            embedding_dim=self.embedding_dim
        )
        model.eval()

        dummy_input = torch.randn(1, self.input_length)

        torch.onnx.export(
            model,
            dummy_input,
            self.onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            external_data=False,
        )

    def encode(self, sequence: np.ndarray) -> List[float]:
        """
        将归一化后的序列编码为128维向量

        Args:
            sequence: 归一化后的numpy数组，长度应与input_length匹配

        Returns:
            128维嵌入向量列表
        """
        if len(sequence) != self.input_length:
            raise ValueError(
                f"序列长度 {len(sequence)} 与模型期望长度 {self.input_length} 不匹配"
            )

        input_tensor = sequence.astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        embedding = outputs[0][0]

        return embedding.tolist()

    def get_embedding_dim(self) -> int:
        """返回嵌入向量维度"""
        return self.embedding_dim
