"""
向量存储 —— ChromaDB 封装，文档向量持久化到硬盘
"""
import chromadb
from chromadb.config import Settings
from src.core.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.core.exceptions import VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """管理文档向量的存储和检索"""

    def __init__(self):
        try:
            self._client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError("ChromaDB 连接或集合创建失败", details={"path": CHROMA_PERSIST_DIR}, cause=e)

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]):
        """把文档块和对应的向量存入 ChromaDB。"""
        try:
            if self._collection.count() > 0:
                self._collection.delete(where={})
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=[chunk["content"] for chunk in chunks],
                metadatas=[
                    {"title": chunk["title"], "chunk_index": chunk["chunk_index"]}
                    for chunk in chunks
                ],
            )
        except Exception as e:
            raise VectorStoreError("文档向量写入失败", details={"chunk_count": len(chunks)}, cause=e)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """用问题向量检索最相关的 top_k 篇文档"""
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                docs.append({
                    "title": f"{meta['title']} (第{meta['chunk_index']+1}段)",
                    "content": results["documents"][0][i],
                    "score": round(1.0 - results["distances"][0][i], 4),
                })
            return docs
        except Exception as e:
            raise VectorStoreError("向量检索失败", details={"top_k": top_k}, cause=e)

    @property
    def is_empty(self) -> bool:
        """检查集合是否为空——决定是否需要导入文档"""
        return self._collection.count() == 0
