"""
嵌入器 —— 使用 FlagEmbedding 加载 BGE-M3，同时输出稠密+稀疏向量
面试要点：BGE-M3 一遍推理同时输出稠密向量（语义）和稀疏向量（关键词）。
           FlagEmbedding 是 BGE 系列模型的官方工具库。
"""
from modelscope import snapshot_download
from FlagEmbedding import BGEM3FlagModel
from src.core.config import EMBED_MODEL_NAME


class Embedder:
    """文字 → 稠密向量 + 稀疏向量（一次推理，两个输出）"""

    def __init__(self):
        # 首次运行从 ModelScope 下载模型到本地
        model_dir = snapshot_download(EMBED_MODEL_NAME, cache_dir="./models")
        # BGEM3FlagModel 专为 BGE-M3 设计，原生支持稠密+稀疏双输出
        self._model = BGEM3FlagModel(model_dir, use_fp16=True)

    # ===== 稠密向量 =====

    def encode(self, texts: list[str]) -> list[list[float]]:
        """输入文字列表，返回稠密向量列表。每个向量 1024 维。"""
        output = self._model.encode(texts, return_dense=True, return_sparse=False)
        return output["dense_vecs"].tolist()

    # ===== 稀疏向量 =====

    def encode_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        """
        输入文字列表，返回稀疏向量列表。
        稀疏向量 = {"token_id": weight, ...} ，稀有词权重高，常见词权重低。
        BGEM3FlagModel 返回 defaultdict(int, {str: np.float32})，转成普通 dict。
        """
        output = self._model.encode(texts, return_dense=False, return_sparse=True)
        return [{str(k): float(v) for k, v in sv.items()} for sv in output["lexical_weights"]]

    # ===== 一次推理同时获取两个输出 =====

    def encode_both(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        """
        一次推理，同时返回稠密向量和稀疏向量。
        BGEM3FlagModel 原生支持，不会增加额外计算量。
        """
        output = self._model.encode(texts, return_dense=True, return_sparse=True)
        dense = output["dense_vecs"].tolist()
        sparse = [{str(k): float(v) for k, v in sv.items()} for sv in output["lexical_weights"]]
        return dense, sparse
