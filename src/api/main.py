"""
SmartKB —— 企业级智能知识库问答系统 API 入口
Author: 曹骏
启动: uvicorn src.api.main:app --reload
"""
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.routes import router, _PUBLIC_PATHS
from src.core.exceptions import SmartKBError, MissingAPIKeyError
from src.utils.logger import get_logger, set_trace_id

logger = get_logger(__name__)

app = FastAPI(
    title="SmartKB",
    description="企业级智能知识库问答系统 - 从RAG到Agentic RAG的完整演进",
    version="0.2.0",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_and_trace_middleware(request: Request, call_next):
    """每次请求：trace_id + JWT 验证（公开路径跳过）"""
    set_trace_id()
    start = time.time()

    # JWT 验证——公开路径跳过
    if request.url.path not in _PUBLIC_PATHS:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            response = JSONResponse(
                status_code=401,
                content={"detail": "缺少 Authorization 头，请先 POST /login 获取 token"},
            )
        else:
            try:
                from src.core.auth import verify_token, get_token_from_header
                token = get_token_from_header(auth_header)
                verify_token(token)
                # token 有效，放行
                response = await call_next(request)
            except ValueError as e:
                response = JSONResponse(status_code=401, content={"detail": str(e)})
            except Exception:
                response = JSONResponse(status_code=401, content={"detail": "token 无效或已过期，请重新登录"})
    else:
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


@app.exception_handler(SmartKBError)
async def smartkb_exception_handler(request: Request, exc: SmartKBError):
    """SmartKB 自定义异常 → HTTP 响应"""
    status_map = {
        "MissingAPIKeyError": 500,
        "ConfigurationError": 500,
        "ModelError": 503,
        "RetrievalError": 502,
        "ValidationError": 422,
    }
    status = status_map.get(exc.__class__.__name__, 500)
    return JSONResponse(
        status_code=status,
        content={"detail": exc.message, "error_type": exc.__class__.__name__},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.include_router(router)
