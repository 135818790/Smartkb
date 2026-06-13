"""
结构化日志 —— 企业级日志规范
面试要点：结构化日志 = 机器可解析 + trace_id 串联全链路 + 分级别输出

使用方式:
  from src.utils.logger import get_logger
  logger = get_logger(__name__)

  logger.info("document_loaded", extra={"file": "config.md", "chunks": 3})
  logger.warning("slow_query", extra={"duration_ms": 2500, "query": "..."})
  logger.error("embedding_failed", extra={"error": str(e), "retry": 2})
"""
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from pythonjsonlogger import jsonlogger


# trace_id 上下文变量 —— 每个请求一个，自动贯穿所有日志
_trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")

# request_id 上下文变量
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class _CustomJsonFormatter(jsonlogger.JsonFormatter):
    """JSON 格式日志，每条日志一行，可被 ELK/Loki 直接索引"""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # 统一时间格式
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        # 日志级别
        log_record["level"] = record.levelname
        # 链路追踪 ID（从上下文变量获取）
        trace_id = _trace_id_ctx.get()
        if trace_id:
            log_record["trace_id"] = trace_id
        request_id = _request_id_ctx.get()
        if request_id:
            log_record["request_id"] = request_id
        # 代码位置
        log_record["logger"] = record.name
        log_record["line"] = record.lineno


def _setup_root_logger():
    """配置根日志器：输出到 stdout（JSON 格式）"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_CustomJsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取日志器。name 通常传 __name__"""
    return logging.getLogger(name)


def set_trace_id(trace_id: str = None, request_id: str = None):
    """设置当前请求的 trace_id。FastAPI middleware 自动调用"""
    if trace_id is None:
        trace_id = str(uuid.uuid4())[:8]
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    _trace_id_ctx.set(trace_id)
    _request_id_ctx.set(request_id)


def get_trace_id() -> str:
    return _trace_id_ctx.get() or "-"


# 模块加载时自动配置
_setup_root_logger()
