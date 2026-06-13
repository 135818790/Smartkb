"""
精排器 —— BGE-Reranker Cross-Encoder，对粗筛结果逐条精确打分
面试必问：双塔模型（BGE-M3）快但粗 → Cross-Encoder 慢但准。
策略：粗筛 Top-20 → 精排 Top-5。速度精度平衡。
"""
from modelscope import snapshot_download
from FlagEmbedding import FlagReranker
from src.core.config import RERANKER_MODEL_NAME, RERANKER_MODEL_DIR, TOP_K


class Reranker:
    """Cross-Encoder 精排——输入 (问题, 文档) pair，输出精确相关分"""

    def __init__(self):
        # 首次运行从 ModelScope 下载（约 1GB）
        model_dir = snapshot_download(RERANKER_MODEL_NAME, cache_dir="./models")
        self._model = FlagReranker(model_dir, use_fp16=True)

    def rerank(self, query: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
        """
        对粗筛候选列表逐条精排。
        Cross-Encoder 把 (问题, 文档) 拼成一对输入，逐 token 交叉注意力打分。
        精度比双塔高，但速度慢。所以只对粗筛结果的 Top-N 使用。
        """
        if not candidates:
            return []

        # 构建 (query, doc) 对
        pairs = [(query, c["content"]) for c in candidates]

        # Cross-Encoder 一次性批量打分
        scores = self._model.compute_score(pairs, normalize=True)

        # 按新分数重排
        for i, c in enumerate(candidates):
            c["score"] = round(float(scores[i]), 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
