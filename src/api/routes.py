"""
API 路由 —— 对外暴露的 HTTP 接口
Step 6：用 LangGraph Agent 替代手写管线
面试要点：从"手写检索→生成"升级为"状态机 Agent"——Router→Retrieve→Generate→Verify
"""
from openai import OpenAI
from fastapi import APIRouter
from pydantic import BaseModel

from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from src.rag.document_loader import load_documents
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.sparse_index import SparseIndex
from src.rag.reranker import Reranker
from src.rag.chunker import chunk_document
from src.agent.graph import build_graph
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

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
        logger.info("loading_models_start")
        _embedder = Embedder()
        _store = VectorStore()
        _sparse_index = SparseIndex()
        _reranker = Reranker()
        logger.info("models_loaded", extra={"embedder": "BGE-M3", "reranker": "bge-reranker-v2-m3"})

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


# --- 请求/响应模型 ---

class ChatRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    title: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


# --- 接口 ---

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """核心接口：用户问问题，系统返回答案 + 引用来源"""
    _ensure_loaded()

    initial_state = {
        "question": req.question,
        "query_dense": _embedder.encode([req.question])[0],
        "query_sparse": _embedder.encode_sparse([req.question])[0],
        "question_type": "",
        "refined_query": "",
        "hits": [],
        "answer": "",
        "verification": "",
        "retry_count": 0,
    }

    final_state = _agent_graph.invoke(initial_state)

    answer = final_state.get("answer", "")
    hits = final_state.get("hits", [])

    logger.info(
        "chat_completed",
        extra={
            "question": req.question[:80],
            "question_type": final_state.get("question_type", ""),
            "verification": final_state.get("verification", ""),
            "retry_count": final_state.get("retry_count", 0),
            "hits_count": len(hits),
        },
    )

    sources = [
        SourceDoc(title=h["title"], score=h["score"], snippet=h["content"][:500])
        for h in hits
    ]
    return ChatResponse(answer=answer, sources=sources)


@router.get("/health")
def health():
    return {"status": "ok"}
