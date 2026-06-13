"""
稀疏向量索引 —— BGE-M3 稀疏输出 = 词汇级精确匹配
面试要点：稠密向量不认识"E10023"，稀疏向量认识。两者互补，不是替代。
"""


class SparseIndex:
    """
    存储 BGE-M3 稀疏向量，支持词汇级检索。
    稀疏向量格式: [{"token": weight, ...}, ...]  （每个文档一个字典）

    面试能说：BGE-M3 一个模型同时输出稠密+稀疏向量。
    稀疏向量本质是学习出来的词权重——和 BM25 同效果，
    但不需要额外部署 Elasticsearch。
    """

    def __init__(self):
        self._doc_vectors: list[dict[str, float]] = []  # 每个文档块的稀疏向量
        self._doc_chunks: list[dict] = []                # 对应的文档块信息
        self._vocab: dict[str, int] = {}                 # 词 → 全局ID（快速查）

    def add(self, chunks: list[dict], sparse_vectors: list[dict[str, float]]):
        """把文档块和对应的稀疏向量存入索引"""
        self._doc_chunks = chunks
        self._doc_vectors = sparse_vectors

        # 建词表（用于后续向量化）
        idx = 0
        for sv in sparse_vectors:
            for token in sv:
                if token not in self._vocab:
                    self._vocab[token] = idx
                    idx += 1

    def search(self, query_sparse: dict[str, float], top_k: int = 20) -> list[dict]:
        """
        用问题的稀疏向量搜索最相关的文档块。
        本质：query_sparse 和每个 doc_sparse 做点积。
        """
        if not self._doc_vectors:
            return []

        scores = []
        for i, doc_sv in enumerate(self._doc_vectors):
            # 点积：只乘两边的交集词（稀疏向量的优势——计算量小）
            score = 0.0
            for token, q_weight in query_sparse.items():
                if token in doc_sv:
                    score += q_weight * doc_sv[token]
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in scores[:top_k]:
            if score > 0:  # 过滤完全不相关的
                title = self._doc_chunks[i]["title"]
                chunk_idx = self._doc_chunks[i].get("chunk_index", 0)
                results.append({
                    "title": f"{title} (第{chunk_idx+1}段)",
                    "content": self._doc_chunks[i]["content"],
                    "score": round(score, 4),
                })
        return results

    @property
    def is_empty(self) -> bool:
        return len(self._doc_vectors) == 0
