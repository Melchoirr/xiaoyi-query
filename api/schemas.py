"""
Pydantic数据模型 - API请求和响应定义
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """数据摄入请求"""

    history_x: List[float] = Field(
        ...,
        description="历史时序数据，长度应与模型期望长度匹配",
        min_length=1
    )
    future_y: List[float] = Field(
        ...,
        description="对应的未来时序数据",
        min_length=1
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "history_x": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
                "future_y": [6.0, 7.0, 8.0, 9.0, 10.0]
            }
        }
    }


class IngestResponse(BaseModel):
    """数据摄入响应"""

    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果描述")
    vector_id: int = Field(..., description="存储向量的ID")
    embedding_dim: int = Field(..., description="嵌入向量维度")


class PredictRequest(BaseModel):
    """预测请求"""

    history_x: List[float] = Field(
        ...,
        description="用于预测的历史时序数据",
        min_length=1
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="检索的最近邻数量"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "history_x": [5.0, 6.0, 7.0, 8.0, 9.0] * 20,
                "top_k": 5
            }
        }
    }


class SearchResultItem(BaseModel):
    """单个检索结果项"""

    id: int = Field(..., description="向量ID")
    score: float = Field(..., description="相似度分数")
    future_y: List[float] = Field(..., description="检索到的未来序列")


class PredictResponse(BaseModel):
    """预测响应"""

    success: bool = Field(..., description="操作是否成功")
    prediction: List[float] = Field(..., description="最终预测的未来序列")
    search_results: List[SearchResultItem] = Field(
        default_factory=list,
        description="检索到的候选结果"
    )
    message: str = Field(default="", description="结果描述")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="服务状态")
    collection_name: str = Field(..., description="向量集合名称")
    vector_count: int = Field(..., description="向量数量")
    embedding_dim: int = Field(..., description="嵌入向量维度")


class StatsResponse(BaseModel):
    """统计信息响应"""

    total_vectors: int = Field(..., description="总向量数")
    collection_info: dict = Field(..., description="集合详细信息")
