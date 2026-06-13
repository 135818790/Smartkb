"""
向量存储 —— ChromaDB 封装，文档向量持久化到硬盘
面试要点：MVP 用 ChromaDB（零运维成本），生产可切 Milvus（支撑百万级文档）
"""
import chromadb
from chromadb.config import Settings
from src.core.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class VectorStore:
    """管理文档向量的存储和检索"""

    def __init__(self):
        # 持久化模式：数据存到 data/vector_store/ 目录
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        # 获取或创建集合（类似数据库的表）
        # hnsw:space=cosine → 用余弦相似度，BGE-M3 的默认度量
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]):
        """把文档块和对应的向量存入 ChromaDB。chunks = chunk_document() 的输出"""
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

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """用问题向量检索最相关的 top_k 篇文档"""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # ChromaDB 返回的是嵌套列表，取第一层（因为只搜了一个问题）
        docs = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            docs.append({
                "title": f"{meta['title']} (第{meta['chunk_index']+1}段)",
                "content": results["documents"][0][i],
                "score": round(1.0 - results["distances"][0][i], 4),
            })
        return docs

    @property
    def is_empty(self) -> bool:
        """检查集合是否为空——决定是否需要导入文档"""
        return self._collection.count() == 0
