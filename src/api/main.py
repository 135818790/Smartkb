"""
SmartKB —— 企业级智能知识库问答系统 API 入口
Author: 曹骏
启动: uvicorn src.api.main:app --reload
"""
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.utils.logger import get_logger, set_trace_id

logger = get_logger(__name__)

app = FastAPI(
    title="SmartKB",
    description="企业级智能知识库问答系统 - 从RAG到Agentic RAG的完整演进",
    version="0.1.0",
)

# CORS 中间件 —— 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """每次请求自动注入 trace_id，记录耗时"""
    set_trace_id()
    start = time.time()

    response = await call_next(request)

    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


app.include_router(router)
