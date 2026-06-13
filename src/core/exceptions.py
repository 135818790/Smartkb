"""
SmartKB 自定义异常层级
面试要点：分层异常 → 每层有明确语义 → 上层捕获后决定重试/降级/告警/返回

使用方式:
  from src.core.exceptions import RetrievalError, ModelNotReadyError
  raise RetrievalError("ChromaDB query timeout", details={"query": "...", "timeout_ms": 5000})
"""
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 异常层级（从基础到具体）
# ============================================================

class SmartKBError(Exception):
    """SmartKB 所有异常的基类"""
    def __init__(self, message: str, details: dict = None, cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        logger.error(
            self.__class__.__name__,
            extra={"message": message, "details": self.details, "cause": str(cause) if cause else None},
        )


# --- 模型层异常 ---

class ModelError(SmartKBError):
    """模型加载/推理失败"""

class ModelNotReadyError(ModelError):
    """模型尚未加载完成"""

class EmbeddingError(ModelError):
    """嵌入向量计算失败"""

class GenerationError(ModelError):
    """LLM 生成失败"""


# --- 检索层异常 ---

class RetrievalError(SmartKBError):
    """检索操作失败"""

class VectorStoreError(RetrievalError):
    """ChromaDB 操作失败"""

class SparseIndexError(RetrievalError):
    """稀疏索引操作失败"""

class RerankerError(RetrievalError):
    """Reranker 精排失败"""

class EmptyRetrievalError(RetrievalError):
    """检索结果为空（非致命）"""


# --- 文档层异常 ---

class DocumentError(SmartKBError):
    """文档处理失败"""

class DocumentLoadError(DocumentError):
    """文档加载/解析失败"""

class ChunkingError(DocumentError):
    """文档分块失败"""


# --- 配置层异常 ---

class ConfigurationError(SmartKBError):
    """配置错误"""

class MissingAPIKeyError(ConfigurationError):
    """API Key 未配置"""


# --- API 层异常 ---

class APIError(SmartKBError):
    """API 请求处理失败"""

class ValidationError(APIError):
    """输入参数校验失败"""

class RateLimitError(APIError):
    """API 调用频率超限"""
