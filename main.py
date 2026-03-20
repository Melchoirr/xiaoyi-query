"""
时序RAG预测微服务 - FastAPI入口
"""

from typing import Generator
from contextlib import asynccontextmanager

import numpy as np

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.processor import TSProcessor
from components.encoder import ONNXEncoder
from components.retriever import QdrantRetriever
from components.ranker import FusionRanker
from api.schemas import (
    IngestRequest,
    IngestResponse,
    PredictRequest,
    PredictResponse,
    HealthResponse,
    StatsResponse,
    SearchResultItem
)
from api.routes_agent import router as agent_router


COLLECTION_NAME = "time_series_rag"
INPUT_LENGTH = 100
STORAGE_PATH = "./qdrant_data"
ONNX_PATH = "encoder_v1.onnx"


processor_instance: TSProcessor = None
encoder_instance: ONNXEncoder = None
retriever_instance: QdrantRetriever = None
ranker_instance: FusionRanker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global processor_instance, encoder_instance, retriever_instance, ranker_instance

    processor_instance = TSProcessor()
    encoder_instance = ONNXEncoder(onnx_path=ONNX_PATH, input_length=INPUT_LENGTH)
    retriever_instance = QdrantRetriever(storage_path=STORAGE_PATH)
    ranker_instance = FusionRanker()

    retriever_instance.create_collection_if_not_exists(
        collection_name=COLLECTION_NAME,
        vector_size=encoder_instance.get_embedding_dim()
    )

    # 将已初始化的 Layer-1 单例注入 Agent Router
    from api import routes_agent
    routes_agent.inject_layer1_instances(
        processor=processor_instance,
        encoder=encoder_instance,
        retriever=retriever_instance,
        ranker=ranker_instance,
    )

    yield

    processor_instance = None
    encoder_instance = None
    retriever_instance = None
    ranker_instance = None


app = FastAPI(
    title="时序RAG预测微服务",
    description="基于向量检索和IDW融合的时序预测服务",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Layer 2: Agentic Router
app.include_router(agent_router)


def get_processor() -> TSProcessor:
    """获取处理器实例"""
    return processor_instance


def get_encoder() -> ONNXEncoder:
    """获取编码器实例"""
    return encoder_instance


def get_retriever() -> QdrantRetriever:
    """获取检索器实例"""
    return retriever_instance


def get_ranker() -> FusionRanker:
    """获取排序器实例"""
    return ranker_instance


@app.get("/", tags=["root"])
async def root():
    """根路径"""
    return {
        "service": "时序RAG预测微服务",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(
    retriever: QdrantRetriever = Depends(get_retriever),
    encoder: ONNXEncoder = Depends(get_encoder)
):
    """健康检查接口"""
    try:
        vector_count = retriever.count_points(COLLECTION_NAME)
        return HealthResponse(
            status="healthy",
            collection_name=COLLECTION_NAME,
            vector_count=vector_count,
            embedding_dim=encoder.get_embedding_dim()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["system"])
async def get_stats(
    retriever: QdrantRetriever = Depends(get_retriever)
):
    """获取统计信息"""
    try:
        info = retriever.get_collection_info(COLLECTION_NAME)
        return StatsResponse(
            total_vectors=info.get("vectors_count", 0),
            collection_info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["data"])
async def ingest_data(
    request: IngestRequest,
    processor: TSProcessor = Depends(get_processor),
    encoder: ONNXEncoder = Depends(get_encoder),
    retriever: QdrantRetriever = Depends(get_retriever)
):
    """
    摄入时序数据到向量数据库

    - 接收history_x和future_y
    - 对history_x进行归一化
    - 通过ONNX编码器生成向量
    - 存储到Qdrant，包含payload
    """
    try:
        if len(request.history_x) != INPUT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"history_x长度必须为{INPUT_LENGTH}"
            )

        normalized, mu, sigma = processor.normalize(request.history_x)

        embedding = encoder.encode(normalized)

        payload = {
            "future_y": request.future_y,
            "mu": mu,
            "sigma": sigma,
            "history_x": request.history_x
        }

        vector_id = retriever.ingest_batch(
            collection_name=COLLECTION_NAME,
            vectors=[embedding],
            payloads=[payload]
        )

        return IngestResponse(
            success=True,
            message="数据摄入成功",
            vector_id=0,
            embedding_dim=encoder.get_embedding_dim()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(
    request: PredictRequest,
    processor: TSProcessor = Depends(get_processor),
    encoder: ONNXEncoder = Depends(get_encoder),
    retriever: QdrantRetriever = Depends(get_retriever),
    ranker: FusionRanker = Depends(get_ranker)
):
    """
    基于检索的时序预测

    - 接收history_x作为查询
    - 归一化并编码为向量
    - 在Qdrant中检索Top-K相似记录
    - 使用IDW融合算法生成预测
    - 反归一化得到真实量级预测
    """
    try:
        if len(request.history_x) != INPUT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"history_x长度必须为{INPUT_LENGTH}"
            )

        normalized, mu, sigma = processor.normalize(request.history_x)

        query_vector = encoder.encode(normalized)

        search_results = retriever.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            top_k=request.top_k
        )

        if not search_results:
            return PredictResponse(
                success=True,
                prediction=[mu] * len(request.history_x[-10:]),
                search_results=[],
                message="未找到相似记录，返回均值"
            )

        future_length = len(search_results[0]["payload"].get("future_y", []))
        if future_length == 0:
            future_length = 10

        fused_prediction_normalized = ranker.rank(
            search_results=search_results,
            future_length=future_length
        )

        denormalized_prediction = processor.denormalize(
            sequence=fused_prediction_normalized,
            mu=mu,
            sigma=sigma
        )

        search_result_items = [
            SearchResultItem(
                id=r["id"],
                score=r["score"],
                future_y=r["payload"].get("future_y", [])
            )
            for r in search_results
        ]

        return PredictResponse(
            success=True,
            prediction=denormalized_prediction.tolist(),
            search_results=search_result_items,
            message=f"基于{len(search_results)}个相似记录预测"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
