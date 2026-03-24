from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import event_router

# ==============================================================================
# FastAPI 应用主入口 (App Entry Point)
# ==============================================================================

app = FastAPI(
    title="Xiaoyi Query Backend Service",
    description="Backend service for Xiaoyi Query, providing news search via Tavily API.",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 路径
    redoc_url="/redoc" # ReDoc 路径
)

# ==============================================================================
# CORS 配置 (Cross-Origin Resource Sharing)
# ==============================================================================
# 允许前端跨域请求
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    # 也可以使用 "*" 允许所有来源 (开发环境方便，生产环境需谨慎)
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 允许所有 HTTP 方法 (GET, POST, etc.)
    allow_headers=["*"], # 允许所有 HTTP 头
)

# ==============================================================================
# 路由注册 (Router Registration)
# ==============================================================================
# 将 event_router 挂载到主应用
app.include_router(event_router.router)

# ==============================================================================
# 健康检查 Endpoint
# ==============================================================================
@app.get("/", tags=["Health"])
async def root():
    """服务的根路径，用于简单的存活检查"""
    return {"message": "Xiaoyi Query Backend is running", "status": "ok"}
