"""
嵌入器 —— 使用 FlagEmbedding 加载 BGE-M3，同时输出稠密+稀疏向量
面试要点：BGE-M3 一遍推理同时输出稠密向量（语义）和稀疏向量（关键词）。
"""
from modelscope import snapshot_download
from FlagEmbedding import BGEM3FlagModel
from src.core.config import EMBED_MODEL_NAME
from src.core.exceptions import EmbeddingError, ModelNotReadyError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """文字 → 稠密向量 + 稀疏向量（一次推理，两个输出）"""

    def __init__(self):
        try:
            model_dir = snapshot_download(EMBED_MODEL_NAME, cache_dir="./models")
            self._model = BGEM3FlagModel(model_dir, use_fp16=True)
        except Exception as e:
            raise EmbeddingError("BGE-M3 模型加载失败", details={"model": EMBED_MODEL_NAME}, cause=e)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """输入文字列表，返回稠密向量列表。每个向量 1024 维。"""
        if not texts:
            return []
        try:
            output = self._model.encode(texts, return_dense=True, return_sparse=False)
            return output["dense_vecs"].tolist()
        except Exception as e:
            raise EmbeddingError("稠密向量编码失败", details={"text_count": len(texts)}, cause=e)

    def encode_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        """输入文字列表，返回稀疏向量列表。"""
        if not texts:
            return []
        try:
            output = self._model.encode(texts, return_dense=False, return_sparse=True)
            return [{str(k): float(v) for k, v in sv.items()} for sv in output["lexical_weights"]]
        except Exception as e:
            raise EmbeddingError("稀疏向量编码失败", details={"text_count": len(texts)}, cause=e)

    def encode_both(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        """一次推理，同时返回稠密向量和稀疏向量。"""
        if not texts:
            return [], []
        try:
            output = self._model.encode(texts, return_dense=True, return_sparse=True)
            dense = output["dense_vecs"].tolist()
            sparse = [{str(k): float(v) for k, v in sv.items()} for sv in output["lexical_weights"]]
            return dense, sparse
        except Exception as e:
            raise EmbeddingError("双向量编码失败", details={"text_count": len(texts)}, cause=e)
