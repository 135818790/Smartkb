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
        print("正在加载 BGE-M3 嵌入模型...")
        _embedder = Embedder()
        _store = VectorStore()
        _sparse_index = SparseIndex()
        print("正在加载 BGE-Reranker 精排模型...")
        _reranker = Reranker()

        # Step 6：创建 LLM 客户端（供 Agent 内部 Router/Generator/Verifier 使用）
        _llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        # 构建 LangGraph Agent
        print("正在构建 LangGraph Agent...")
        _agent_graph = build_graph(
            llm_client=_llm_client,
            embedder=_embedder,
            store=_store,
            sparse_index=_sparse_index,
            reranker=_reranker,
        )
        print("Agent 构建完成")

        if _store.is_empty:
            print("向量库为空，正在导入文档...")
            documents = load_documents()

            all_chunks = []
            for doc in documents:
                chunks = chunk_document(doc)
                all_chunks.extend(chunks)
                print(f"  {doc['title']}: {len(chunks)} 块")

            chunk_texts = [c["content"] for c in all_chunks]

            print(f"正在为 {len(all_chunks)} 个块计算稠密+稀疏向量...")
            dense_embeddings, sparse_vectors = _embedder.encode_both(chunk_texts)

            _store.add_documents(all_chunks, dense_embeddings)
            _sparse_index.add(all_chunks, sparse_vectors)

            print(f"已导入 {len(documents)} 篇文档 → {len(all_chunks)} 个块")
            print(f"  稠密路径: ChromaDB ({len(dense_embeddings)} 条)")
            print(f"  稀疏路径: 内存索引 ({len(sparse_vectors)} 条, {len(_sparse_index._vocab)} 词)")
        else:
            print(f"ChromaDB 已有 {_store._collection.count()} 条稠密向量，跳过导入")
            if _sparse_index.is_empty:
                print("稀疏索引为空，从 ChromaDB 重建...")
                documents = load_documents()
                all_chunks = []
                for doc in documents:
                    chunks = chunk_document(doc)
                    all_chunks.extend(chunks)
                chunk_texts = [c["content"] for c in all_chunks]
                _, sparse_vectors = _embedder.encode_both(chunk_texts)
                _sparse_index.add(all_chunks, sparse_vectors)
                print(f"稀疏索引已重建 ({len(sparse_vectors)} 条)")


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
    """
    Step 6 核心接口：
    从手写 pipeline → LangGraph Agent 状态机。
    Router 判断问题类型 → Retrieve → Generate → Verify（Self-RAG）
    """
    _ensure_loaded()

    # Step 6：构造初始状态，交给 Agent 状态机执行
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

    # 状态机执行——Agent 自己决定 Router→Retrieve→Generate→Verify 的流转
    final_state = _agent_graph.invoke(initial_state)

    # 从最终状态提取结果
    answer = final_state.get("answer", "")
    hits = final_state.get("hits", [])

    sources = [
        SourceDoc(title=h["title"], score=h["score"], snippet=h["content"][:500])
        for h in hits
    ]
    return ChatResponse(answer=answer, sources=sources)


@router.get("/health")
def health():
    return {"status": "ok"}
