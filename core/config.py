"""
core/config.py
===============
全局配置中心 — 所有模块共享的常量和配置项。

优先级（从高到低）：
    1. 环境变量（支持生产环境动态注入）
    2. .env 文件（本地开发使用 python-dotenv 加载）
    3. 本文件中的默认值（硬编码兜底）

使用方式：
    from core.config import cfg
    print(cfg.llm.api_key)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal


# ---------------------------------------------------------------------------
# 路径常量（项目根目录）
# ---------------------------------------------------------------------------
_ROOT_DIR: Path = Path(__file__).parent.parent.resolve()
MODEL_DIR: Path = _ROOT_DIR / "models"
STORAGE_DIR: Path = _ROOT_DIR / "qdrant_data"
ETT_DATA_DIR: Path = _ROOT_DIR / "ETT_data"


# ---------------------------------------------------------------------------
# 尝试验证 .env 文件（如果存在则加载）
# ---------------------------------------------------------------------------
_env_file = _ROOT_DIR / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# LLM Provider 配置
# ---------------------------------------------------------------------------

LLMProvider = Literal["openai", "deepseek", "qwen", "ollama"]


@dataclass
class LLMConfig:
    """
    LLM 推理端点配置。

    支持的 Provider：
        - openai  : OpenAI 官方 API (api.openai.com)，模型如 gpt-4o-mini
        - deepseek: DeepSeek 官方 API (api.deepseek.com)
        - qwen    : 通义千问 API (dashscope.aliyuncs.com)
        - ollama  : 本地 Ollama 服务 (localhost:11434)
    """

    provider: LLMProvider = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # 自定义 API Base URL
    timeout: float = 30.0
    temperature: float = 0.0

    # Provider-specific defaults
    _provider_defaults: dict = field(default_factory=lambda: {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "qwen-plus",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama3.2",
        },
    }, repr=False, compare=False)

    def __post_init__(self) -> None:
        defaults = self._provider_defaults.get(self.provider, {})
        if self.base_url is None:
            self.base_url = defaults.get("base_url", "")
        if self.model == "gpt-4o-mini" and self.provider != "openai":
            self.model = defaults.get("default_model", self.model)
        if self.api_key is None:
            self.api_key = os.environ.get(f"{self.provider.upper()}_API_KEY") \
                or os.environ.get("OPENAI_API_KEY") \
                or os.environ.get("DEEPSEEK_API_KEY") \
                or os.environ.get("QWEN_API_KEY") \
                or os.environ.get("API_KEY")

    def to_dict(self) -> dict:
        """返回可序列化的字典（不包含 api_key）。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "temperature": self.temperature,
        }


# ---------------------------------------------------------------------------
# 系统级配置
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """系统级常量和路径配置。"""

    # --- 路径 ---
    root_dir: Path = _ROOT_DIR
    model_dir: Path = MODEL_DIR
    storage_dir: Path = STORAGE_DIR
    ett_data_dir: Path = ETT_DATA_DIR

    # --- 向量数据库 ---
    qdrant_storage_path: str = "./qdrant_data"
    qdrant_collection: str = "time_series_rag"
    qdrant_distance_metric: str = "Cosine"  # Cosine | Euclidean | Dot

    # --- 时序处理 ---
    input_length: int = 100          # 历史窗口长度 L
    default_future_length: int = 48  # 默认预测窗口长度
    default_batch_size: int = 500

    # --- 编码器 ---
    onnx_encoder_path: str = "models/foundation_encoder.onnx"
    # TS2Vec ONNX 规格（由 export_foundation_model.py 导出）
    ts2vec_seq_len: int = 512       # TS2Vec 固定输入长度（超过截断，不足补零）
    ts2vec_embedding_dim: int = 320  # TS2Vec 隐层维度

    # --- 精排模型 ---
    xgb_ranker_path: str = "models/xgb_ranker.json"

    def get_encoder_path(self) -> Path:
        p = self.root_dir / self.onnx_encoder_path
        if not p.exists():
            p = self.root_dir / "encoder_v1.onnx"  # 兼容旧路径
        return p

    def get_ranker_path(self) -> Path:
        return self.root_dir / self.xgb_ranker_path


# ---------------------------------------------------------------------------
# 全局配置单例
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    全局配置单例，整合所有子配置。

    使用示例：
        from core.config import cfg
        cfg.system.input_length
        cfg.llm.provider
        cfg.llm.model
    """

    system: SystemConfig = field(default_factory=SystemConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_json(self, path: Optional[Path] = None) -> str:
        """将当前配置序列化为 JSON 字符串（不含 API Key）。"""
        data = {
            "system": {
                "input_length": self.system.input_length,
                "default_future_length": self.system.default_future_length,
                "qdrant_collection": self.system.qdrant_collection,
                "qdrant_distance_metric": self.system.qdrant_distance_metric,
                "ts2vec_seq_len": self.system.ts2vec_seq_len,
                "ts2vec_embedding_dim": self.system.ts2vec_embedding_dim,
                "onnx_encoder_path": self.system.onnx_encoder_path,
                "xgb_ranker_path": self.system.xgb_ranker_path,
            },
            "llm": self.llm.to_dict(),
        }
        txt = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            Path(path).write_text(txt, encoding="utf-8")
        return txt


# ---------------------------------------------------------------------------
# 全局单例（模块级实例，供所有消费者 import）
# ---------------------------------------------------------------------------
cfg: Config = Config()


# ---------------------------------------------------------------------------
# 便捷访问器（避免每次写 cfg.xxx）
# ---------------------------------------------------------------------------
def get_system() -> SystemConfig:
    return cfg.system


def get_llm() -> LLMConfig:
    return cfg.llm
