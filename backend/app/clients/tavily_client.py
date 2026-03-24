import os
import httpx
from typing import Optional, List, Dict, Any
from fastapi import HTTPException
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

class TavilyClient:
    """
    专门封装与 Tavily API 交互的逻辑 (Client Layer)
    负责构建请求、发送请求、并处理原始的 API 响应
    """
    
    BASE_URL = "https://api.tavily.com/search"

    def __init__(self):
        # 实际项目中，最好通过依赖注入或者全局配置加载 API Key
        # 这里为了演示，直接从环境变量读取
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            # 如果没有 API Key，抛出异常或仅记录日志（但这会阻碍服务启动）
            raise ValueError("Environment variable 'TAVILY_API_KEY' is not set.")

    async def search(self, query: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        向 Tavily API 发送异步 GET/POST 请求获取新闻数据
        
        Args:
            query (str): 搜索关键词
            start_date (str, optional): 开始日期 ISO 格式
            end_date (str, optional): 结束日期 ISO 格式
            
        Returns:
            List[Dict[str, Any]]: 包含原始新闻对象（字典）的列表
        """
        
        # 构建请求参数 payload
        # Tavily API 标准参数：api_key, query, search_depth, include_images等
        # 这里使用了 search_depth="advanced" 来获取更全面的结果
        # 注意：Tavily 的 API 当前版本对日期过滤的支持可能有限制，
        # 如果 start_date/end_date 不直接支持，通常我会把日期作为 query 的一部分附加
        # 但在演示中，如果 Tavily 的 Python SDK 支持 filter，我们可以在 header/body 里传
        # 为了通用性，此处把日期拼接到 query (Prompt Engineering) 或直接作为参数 (取决于具体 API 文档)
        
        # 假设 Tavily 的 /search 接口接受 POST JSON
        # 参考官方文档：POST https://api.tavily.com/search
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced", # "basic" or "advanced"
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": 5, # 可以按需调整
        }

        # 如果有日期限制，Tavily API 可能不直接支持 start_date 参数，
        # 这里尝试通过 prompt engineering 优化 query，或者如果 API 支持的话加入参数
        # (注：标准的 Tavily API 暂时没有明确的 start/end date 过滤参数，通常在 query 里描述 "news from 2023-01-01 to ...")
        # 为符合题目要求，通过修改 query 来尝试引导结果
        date_context = ""
        if start_date and end_date:
            date_context = f" between {start_date} and {end_date}"
        elif start_date:
            date_context = f" after {start_date}"
        elif end_date:
            date_context = f" before {end_date}"
            
        if date_context:
            payload["query"] = f"{query}{date_context}"

        # 使用 httpx 发送异步请求
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.BASE_URL, json=payload, timeout=10.0)
                response.raise_for_status() # 检查 HTTP 状态码
                
                data = response.json()
                # Tavily 返回结构通常是 {"results": [...], "query": ..., "answer": ...}
                return data.get("results", [])
                
            except httpx.HTTPStatusError as e:
                # 处理 HTTP 错误 (4xx, 5xx)
                print(f"Server error: {e}")
                raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch data from Tavily API")
            except httpx.RequestError as e:
                # 处理连接错误
                print(f"Request error: {e}")
                raise HTTPException(status_code=503, detail="Service unavailable: Could not connect to search provider")
            except Exception as e:
                # 处理其他未知错误
                print(f"Unknown error: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error during search operation")
