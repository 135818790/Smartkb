"""
API 路由 —— 对外暴露的 HTTP 接口
Step 6：用 LangGraph Agent 替代手写管线
面试要点：从"手写检索→生成"升级为"状态机 Agent"——Router→Retrieve→Generate→Verify
"""
from openai import OpenAI
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from src.core.exceptions import (
    EmbeddingError, VectorStoreError, GenerationError,
    DocumentLoadError, ChunkingError, ConfigurationError,
)
from src.core.auth import create_token
from src.rag.document_loader import load_documents
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.sparse_index import SparseIndex
from src.rag.reranker import Reranker
from src.rag.chunker import chunk_document
from src.rag.generator import generate_answer, generate_answer_stream
from src.agent.graph import build_graph
from src.utils.logger import get_logger
from collections import defaultdict
import uuid

logger = get_logger(__name__)
router = APIRouter()

# 公开路径——无需 JWT
_PUBLIC_PATHS = {"/health", "/login", "/docs", "/openapi.json"}

# 全局单例
_embedder = None
_store = None
_sparse_index = None
_reranker = None
_agent_graph = None
_llm_client = None


def _ensure_loaded():
    """懒加载：第一次请求时加载模型 + 构建 LangGraph Agent"""
    global _embedder, _store, _sparse_index, _reranker, _agent_graph, _llm_client
    if _embedder is None:
        try:
            logger.info("loading_models_start")
            _embedder = Embedder()
            _store = VectorStore()
            _sparse_index = SparseIndex()
            _reranker = Reranker()
            logger.info("models_loaded", extra={"embedder": "BGE-M3", "reranker": "bge-reranker-v2-m3"})
        except Exception as e:
            raise EmbeddingError("嵌入模型或向量存储加载失败", details={"error": str(e)}, cause=e)

        _llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        _agent_graph = build_graph(
            llm_client=_llm_client,
            embedder=_embedder,
            store=_store,
            sparse_index=_sparse_index,
            reranker=_reranker,
        )
        logger.info("agent_graph_built")

        if _store.is_empty:
            _import_documents()
        else:
            logger.info("vector_store_non_empty", extra={"count": _store._collection.count()})
            if _sparse_index.is_empty:
                logger.info("sparse_index_rebuilding")
                documents = load_documents()
                all_chunks = []
                for doc in documents:
                    chunks = chunk_document(doc)
                    all_chunks.extend(chunks)
                chunk_texts = [c["content"] for c in all_chunks]
                _, sparse_vectors = _embedder.encode_both(chunk_texts)
                _sparse_index.add(all_chunks, sparse_vectors)
                logger.info("sparse_index_rebuilt", extra={"vectors": len(sparse_vectors)})


def _import_documents():
    """导入文档：加载 → 分块 → 编码 → 存入 ChromaDB 和稀疏索引"""
    logger.info("document_import_start")
    documents = load_documents()

    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        logger.debug("document_chunked", extra={"title": doc["title"], "chunks": len(chunks)})

    chunk_texts = [c["content"] for c in all_chunks]
    dense_embeddings, sparse_vectors = _embedder.encode_both(chunk_texts)

    _store.add_documents(all_chunks, dense_embeddings)
    _sparse_index.add(all_chunks, sparse_vectors)

    logger.info(
        "document_import_done",
        extra={
            "documents": len(documents),
            "chunks": len(all_chunks),
            "dense_count": len(dense_embeddings),
            "sparse_count": len(sparse_vectors),
            "vocab_size": len(_sparse_index._vocab),
        },
    )


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """
    登录获取 JWT token。
    生产环境：验证数据库用户密码。开发环境：接受任意凭据。
    面试说法：JWT 无状态验证，不查数据库，水平扩展无瓶颈。
    """
    # 开发简化版——生产环境替换为数据库验证
    if not req.username or not req.password:
        raise HTTPException(status_code=422, detail="用户名和密码不能为空")
    token = create_token(req.username)
    logger.info("user_logged_in", extra={"username": req.username})
    return LoginResponse(access_token=token)


# --- 会话管理（企业生产用 Redis，MVP 用内存） ---

class SessionStore:
    """多轮对话会话管理。生产环境替换为 Redis"""
    def __init__(self, max_history: int = 10):
        self._sessions: dict[str, list[str]] = defaultdict(list)
        self._max = max_history

    def add(self, session_d: str, question: str, answer: str):
        """保存一轮对话"""
        self._sessions[session_id].append(f"用户: {question}")
        self._sessions[session_id].append(f"助手: {answer}")
        # 保持最近 N 轮
        if len(self._sessions[session_id]) > self._max * 2:
            self._sessions[session_id] = self._sessions[session_id][-(self._max * 2):]

    def get_history(self, session_id: str) -> list[str]:
        """获取最近对话历史"""
        return self._sessions.get(session_id, [])

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)


_sessions = SessionStore()


# --- 请求/响应模型 ---

class ChatRequest(BaseModel):
    question: str
    session_id: str = ""  # 空字符串表示新会话


class SourceDoc(BaseModel):
    title: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    session_id: str = ""  # 返回会话 ID，前端用于多轮对话


# --- 非流式接口 ---

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """核心接口：用户问问题，系统返回答案 + 引用来源（支持多轮对话）"""
    _ensure_loaded()

    # 会话管理：没有 session_id 就创建新的
    session_id = req.session_id or str(uuid.uuid4())[:8]

    # 获取历史对话
    history = _sessions.get_history(session_id)

    initial_state = {
        "question": req.question,
        "query_dense": _embedder.encode([req.question])[0],
        "query_sparse": _embedder.encode_sparse([req.question])[0],
        "question_type": "",
        "refined_query": "",
        "history": history,
        "hits": [],
        "answer": "",
        "verification": "",
        "retry_count": 0,
    }

    final_state = _agent_graph.invoke(initial_state)

    answer = final_state.get("answer", "")
    hits = final_state.get("hits", [])

    # 保存本轮对话
    _sessions.add(session_id, req.question, answer)

    logger.info(
        "chat_completed",
        extra={
            "session_id": session_id,
            "history_len": len(history),
            "question": req.question[:80],
            "question_type": final_state.get("question_type", ""),
            "verification": final_state.get("verification", ""),
            "hits_count": len(hits),
        },
    )

    sources = [
        SourceDoc(title=h["title"], score=h["score"], snippet=h["content"][:500])
        for h in hits
    ]
    return ChatResponse(answer=answer, sources=sources, session_id=session_id)


# --- 流式接口（SSE） ---

from fastapi.responses import StreamingResponse


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    流式接口：Server-Sent Events 逐字返回回答。
    面试说法："用 SSE 而非 WebSocket——SSE 更轻量，单向推送够用，
              HTTP 协议天然支持断线重连，调试也方便。"

    SSE 格式:
      data: {"token": "连接"}   ← 每个 token 一行
      data: {"token": "池"}
      ...
      data: {"done": true, "sources": [...], "session_id": "..."}  ← 结束信号
    """
    import json as _json

    _ensure_loaded()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    # Router + Retrieve：手动调用混合检索（不用 Agent 图，更快）
    from src.rag.retriever import retrieve
    query_dense = _embedder.encode([req.question])[0]
    query_sparse = _embedder.encode_sparse([req.question])[0]
    hits = retrieve(
        query=req.question,
        query_dense=query_dense,
        query_sparse=query_sparse,
        store=_store,
        sparse_index=_sparse_index,
        reranker=_reranker,
    )

    def sse_generator():
        full_answer = ""
        try:
            for token in generate_answer_stream(req.question, hits):
                full_answer += token
                yield f"data: {_json.dumps({'token': token})}\n\n"
            # 保存会话
            _sessions.add(session_id, req.question, full_answer)
            # 结束信号：带来源和 session_id
            sources_data = [
                {"title": h["title"], "score": h["score"]}
                for h in hits
            ]
            yield f"data: {_json.dumps({'done': True, 'sources': sources_data, 'session_id': session_id})}\n\n"
        except Exception as e:
            logger.error("stream_error", extra={"error": str(e)})
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


@router.get("/health")
def health():
    return {"status": "ok"}
