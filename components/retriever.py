"""
Qdrant向量检索引擎 - 本地磁盘模式
"""

from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
)


class QdrantRetriever:
    """Qdrant向量检索器"""

    def __init__(self, storage_path: str = "./qdrant_data"):
        """
        初始化Qdrant客户端

        Args:
            storage_path: 本地存储路径
        """
        self.client = QdrantClient(path=storage_path)

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int = 128,
        distance: Distance = Distance.COSINE
    ) -> None:
        """
        创建集合（如果不存在）

        Args:
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量方式
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )

    def delete_collection(self, collection_name: str) -> None:
        """删除集合"""
        self.client.delete_collection(collection_name=collection_name)

    def ingest_batch(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> int:
        """
        批量写入向量及payload

        Args:
            collection_name: 集合名称
            vectors: 向量列表
            payloads: 每个向量对应的元数据payload

        Returns:
            写入的向量数量
        """
        points = []
        for idx, (vector, payload) in enumerate(zip(vectors, payloads)):
            point = PointStruct(
                id=idx,
                vector=vector,
                payload=payload
            )
            points.append(point)

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

        return len(points)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        搜索最相似的Top-K向量

        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            top_k: 返回的最相似结果数量
            score_threshold: 相似度阈值

        Returns:
            搜索结果列表，每项包含 id, score, payload
        """
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_vectors=False,
            with_payload=True
        )

        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        return results

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取集合信息"""
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "name": info.name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }

    def count_points(self, collection_name: str) -> int:
        """获取集合中的向量数量"""
        result = self.client.count(
            collection_name=collection_name,
            exact=True
        )
        return result.count
