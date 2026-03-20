"""
ONNX 向量化引擎 — 使用预训练时序基础模型 TS2Vec

【模型规格】
    ONNX 模型:  models/foundation_encoder.onnx
                由 scripts/export_foundation_model.py 从 TS2Vec Encoder 导出

    输入张量:
        name:   "input"
        shape:  [batch=1, seq_len=512, channels=1]
                dtype: float32
                注: 低于 512 的序列在头部补零；超过 512 的序列截断尾部

    输出张量:
        name:   "embedding"
        shape:  [batch=1, seq_len=512, embedding_dim=320]
                dtype: float32

【在线推理流程】
    1. 接收归一化后的 1D numpy 数组 sequence (L,)
    2. Zero-padding / Truncation → [512,] 适配 TS2Vec 固定输入长度
    3. Reshape 为 [1, 512, 1] → ONNX 推理 → [1, 512, 320]
    4. Mean Pooling over seq_len 维度 → [1, 1, 320] → squeeze → [320]
    5. L2 Normalize → [320] (单位球面，适配 Qdrant Cosine 距离)
    6. 返回 List[float] (320 维)

【物理隔离保证】
    本模块仅依赖 onnxruntime + numpy，不引入 torch / transformers，
    可安全部署在 FastAPI 线上服务中。
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime


# ---------------------------------------------------------------------------
# 全局常量（与 scripts/export_foundation_model.py 保持一致）
# ---------------------------------------------------------------------------
MODEL_DIR: Path = Path(__file__).parent.parent / "models"
DEFAULT_ONNX_PATH: Path = MODEL_DIR / "foundation_encoder.onnx"

# TS2Vec Encoder 固定输入序列长度（与导出时的 --input-length 一致）
TS2VEC_SEQ_LEN: int = 512

# TS2Vec Encoder 隐层维度（固定输出 feature 维度）
EMBEDDING_DIM: int = 320


class ONNXEncoder:
    """
    预训练时序基础模型 ONNX 推理引擎。

    加载 TS2Vec Encoder ONNX 模型，接收 Z-Score 归一化后的序列，
    输出 320 维 L2 归一化嵌入向量。

    Attributes:
        onnx_path:    ONNX 模型文件路径
        input_length: 期望的历史序列长度（本系统固定为 100）
        embedding_dim: 嵌入向量维度（TS2Vec = 320）
        session:       onnxruntime.InferenceSession 实例
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        input_length: int = 100,
    ):
        """
        初始化 ONNX 推理会话。

        Args:
            onnx_path:    ONNX 模型路径，默认使用 models/foundation_encoder.onnx
            input_length: 期望的历史序列长度（由 TSProcessor 归一化后的输入长度，
                          本系统固定为 100）
        """
        self.onnx_path: Path = Path(onnx_path) if onnx_path else DEFAULT_ONNX_PATH
        self.input_length: int = input_length
        self.embedding_dim: int = EMBEDDING_DIM

        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX 模型文件不存在: {self.onnx_path}\n"
                f"请先运行 scripts/export_foundation_model.py 导出 TS2Vec 模型，"
                f"或确认 models/foundation_encoder.onnx 已放置在正确位置。"
            )

        # ONNX Runtime Session 配置
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # 获取输入 / 输出张量名称
        self._input_name: str = self.session.get_inputs()[0].name
        self._output_name: str = self.session.get_outputs()[0].name

        # 验证 ONNX 模型规格
        self._validate_onnx_model()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def encode(self, sequence: np.ndarray) -> List[float]:
        """
        将归一化后的 1D 序列编码为 320 维嵌入向量。

        流程:
            1. 长度校验
            2. Zero-padding / Truncation → [512]
            3. Reshape → [1, 512, 1]
            4. ONNX 推理 → [1, 512, 320]
            5. Mean Pooling → [1, 1, 320] → squeeze → [320]
            6. L2 归一化 → [320]
            7. 返回 List[float]

        Args:
            sequence: 归一化后的 numpy 1-D 数组，长度应为 input_length (= 100)

        Returns:
            320 维 L2 归一化嵌入向量（Python List[float]）

        Raises:
            ValueError: 输入序列长度与 self.input_length 不匹配
        """
        # ---- 1. 长度校验 ----
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 1:
            raise ValueError(
                f"encode() 仅接受 1-D 数组，实际 ndim={seq.ndim}"
            )
        if seq.shape[0] != self.input_length:
            raise ValueError(
                f"序列长度 {seq.shape[0]} 与模型期望长度 {self.input_length} 不匹配"
            )

        # ---- 2. Zero-padding / Truncation ----
        #   TS2Vec 固定输入长度为 512，低于 512 的序列在头部补零，
        #   超过 512 的序列截取最后 512 个点。
        fixed_seq = self._pad_or_truncate(seq)

        # ---- 3. Reshape → [1, 512, 1] ----
        #   TS2Vec ONNX 输入规格: [batch=1, seq_len=512, channels=1]
        input_tensor: np.ndarray = fixed_seq.reshape(1, TS2VEC_SEQ_LEN, 1)

        # ---- 4. ONNX 推理 ----
        outputs: List[np.ndarray] = self.session.run(
            [self._output_name],
            {self._input_name: input_tensor}
        )
        # outputs[0].shape == [1, 512, 320]
        embeddings: np.ndarray = outputs[0]

        # ---- 5. Mean Pooling over seq_len 维度 ----
        #   对时间维度（dim=1，即 512 个时间步）做平均池化，
        #   得到全局序列表征 [1, 1, 320]。
        pooled: np.ndarray = np.mean(embeddings, axis=1, keepdims=True)
        # squeeze → [320]
        vector: np.ndarray = pooled.squeeze(axis=(0, 1))

        # ---- 6. L2 归一化 ----
        norm: float = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm
        else:
            # 全零向量保护（理论上不会发生）
            vector = np.zeros(self.embedding_dim, dtype=np.float32)

        # ---- 7. 返回 List[float] ----
        return vector.tolist()

    def get_embedding_dim(self) -> int:
        """返回嵌入向量维度（320）"""
        return self.embedding_dim

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _pad_or_truncate(self, seq: np.ndarray) -> np.ndarray:
        """
        将任意长度 L 的序列对齐为 TS2Vec 固定长度 TS2VEC_SEQ_LEN (= 512)。

        策略:
            - L < 512: 头部补零（pre-padding），使长度等于 512
                      理由：TS2Vec 使用因果卷积，头部补零不会产生泄漏的"未来"信息
            - L >= 512: 截取最后 512 个点（tail truncation），
                        保留最近的时间上下文
        """
        target_len: int = TS2VEC_SEQ_LEN
        actual_len: int = seq.shape[0]

        if actual_len < target_len:
            # 头部补零（pre-padding）
            pad_width: int = target_len - actual_len
            padded: np.ndarray = np.pad(
                seq,
                (pad_width, 0),       # (前端补零, 后端不补)
                mode="constant",
                constant_values=0.0,
            )
            return padded
        else:
            # 截取最后 target_len 个点
            return seq[-target_len:]

    def _validate_onnx_model(self) -> None:
        """
        启动时验证 ONNX 模型输入输出规格是否符合 TS2Vec 预期。

        执行一个轻量级推理，校验：
            输入 shape  = [1, 512, 1]
            输出 shape  = [1, 512, 320]
        """
        # 构造最小测试张量（全零）
        test_input: np.ndarray = np.zeros(
            (1, TS2VEC_SEQ_LEN, 1), dtype=np.float32
        )

        test_outputs: List[np.ndarray] = self.session.run(
            [self._output_name],
            {self._input_name: test_input}
        )

        expected_output_shape: tuple = (1, TS2VEC_SEQ_LEN, self.embedding_dim)
        actual_shape: tuple = test_outputs[0].shape

        if actual_shape != expected_output_shape:
            raise ValueError(
                f"ONNX 模型输出 shape 不符合预期。\n"
                f"  期望: {expected_output_shape}\n"
                f"  实际: {actual_shape}\n"
                f"请确认使用的 ONNX 模型是由 TS2Vec Encoder 导出。"
            )
