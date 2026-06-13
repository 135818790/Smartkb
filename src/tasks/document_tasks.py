"""
异步文档处理任务 —— 上传后后台处理，用户不等待
面试说法: "文档上传走 Celery 异步管线——解析→分块→嵌入→入库。
          前端轮询 GET /documents/status/{task_id} 查进度。"
"""
from pathlib import Path
from celery import current_task
from src.tasks.celery_app import celery_app
from src.core.config import DOCUMENTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Celery worker 自己的模型实例（懒加载，不和 API 进程共享）
_worker_embedder = None
_worker_store = None
_worker_sparse_index = None


def _get_worker_components():
    """懒加载模型组件——Celery worker 首次处理任务时初始化"""
    global _worker_embedder, _worker_store, _worker_sparse_index
    if _worker_embedder is None:
        from src.rag.embedder import Embedder
        from src.rag.vector_store import VectorStore
        from src.rag.sparse_index import SparseIndex
        from src.rag.chunker import chunk_document
        from src.rag.document_loader import _read_file

        logger.info("celery_worker_loading_models")
        _worker_embedder = Embedder()
        _worker_store = VectorStore()
        _worker_sparse_index = SparseIndex()
        logger.info("celery_worker_models_loaded")
    return _worker_embedder, _worker_store, _worker_sparse_index


@celery_app.task(bind=True, name="process_document")
def process_document(self, file_path: str, filename: str) -> dict:
    """
    异步处理单个文档文件：
    1. 读取文件内容
    2. 语义分块
    3. 嵌入向量（稠密 + 稀疏）
    4. 存入 ChromaDB + 稀疏索引

    返回: {"status": "success", "chunks": 5, "filename": "xxx.md"}
    """
    from src.rag.chunker import chunk_document

    self.update_state(state="STARTED", meta={"progress": 10, "step": "读取文件"})

    # 步骤 1：读取文件
    filepath = Path(file_path)
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("task_file_read_failed", extra={"file": filename, "error": str(e)})
        return {"status": "failed", "error": f"文件读取失败: {e}"}

    doc = {"title": filepath.stem, "content": content}
    logger.info("task_file_loaded", extra={"file": filename, "size": len(content)})

    self.update_state(state="STARTED", meta={"progress": 30, "step": "语义分块"})

    # 步骤 2：分块
    try:
        chunks = chunk_document(doc)
    except Exception as e:
        logger.error("task_chunking_failed", extra={"file": filename, "error": str(e)})
        return {"status": "failed", "error": f"分块失败: {e}"}

    if not chunks:
        logger.warning("task_empty_chunks", extra={"file": filename})
        return {"status": "skipped", "reason": "文件内容为空"}

    self.update_state(state="STARTED", meta={"progress": 50, "step": f"嵌入 {len(chunks)} 个块"})

    # 步骤 3：获取 worker 自己的模型组件
    embedder, store, sparse_index = _get_worker_components()

    # 步骤 4：嵌入
    chunk_texts = [c["content"] for c in chunks]
    try:
        dense_embeddings, sparse_vectors = embedder.encode_both(chunk_texts)
    except Exception as e:
        logger.error("task_embedding_failed", extra={"file": filename, "error": str(e)})
        return {"status": "failed", "error": f"嵌入失败: {e}"}

    self.update_state(state="STARTED", meta={"progress": 80, "step": "写入向量库"})

    # 步骤 5：存入 ChromaDB + 稀疏索引
    try:
        store.add_documents(chunks, dense_embeddings)
        sparse_index.add(chunks, sparse_vectors)
    except Exception as e:
        logger.error("task_store_failed", extra={"file": filename, "error": str(e)})
        return {"status": "failed", "error": f"向量存储失败: {e}"}

    logger.info("task_document_processed", extra={
        "file": filename, "chunks": len(chunks),
        "dense_count": len(dense_embeddings), "sparse_count": len(sparse_vectors),
    })

    return {
        "status": "success",
        "filename": filename,
        "chunks": len(chunks),
        "message": f"文档 {filename} 处理完成，共 {len(chunks)} 个块",
    }
