"""
混合检索器 —— 稠密 + 稀疏双路召回 → RRF 融合 → Reranker 精排
"""
from src.core.config import TOP_K, HYBRID_TOP_K, RRF_K
from src.rag.vector_store import VectorStore
from src.rag.sparse_index import SparseIndex
from src.rag.reranker import Reranker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve(
    query: str,
    query_dense: list[float],
    query_sparse: dict[str, float],
    store: VectorStore,
    sparse_index: SparseIndex,
    reranker: Reranker,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    混合检索完整管线：
    1. 稠密召回（ChromaDB 余弦检索）
    2. 稀疏召回（词汇级点积匹配）
    3. RRF 融合（倒数排名合并）
    4. BGE-Reranker 精排（Cross-Encoder 逐条打分）
    5. 返回 Top-K

    面试能说：每一步解决一个具体问题。
    """
    # ── 第 1 路：稠密召回 ──
    dense_results = store.search(query_dense, top_k=HYBRID_TOP_K)

    # ── 第 2 路：稀疏召回 ──
    sparse_results = sparse_index.search(query_sparse, top_k=HYBRID_TOP_K)

    # ── 第 3 步：RRF 融合 ──
    merged = _rrf_fusion(dense_results, sparse_results, k=RRF_K)

    # ── 第 4 步：Reranker 精排（可选——兼容性降级）──
    try:
        final = reranker.rerank(query, merged, top_k=top_k)
    except Exception as e:
        logger.warning("reranker_failed_fallback", extra={"error": str(e)})
        final = merged[:top_k]

    return final


def _rrf_fusion(list_a: list[dict], list_b: list[dict], k: int = 60) -> list[dict]:
    """
    倒数排名融合（Reciprocal Rank Fusion）。
    不看绝对分数，只看排名。两路都排前面的文档胜出。

    公式: RRF(doc) = Σ 1/(k + rank_of_doc_in_list)
    k=60 是经验值，让排名差异不那么极端。

    面试能说：
    "稠密分数(0.71)和稀疏分数(0.95)不在同一尺度，不能直接相加。
    RRF 只看排名不看分数，解决了跨检索器的分数校准问题。"
    """
    merged_map = {}  # title → {"doc": ..., "score": float}

    for rank, doc in enumerate(list_a):
        key = doc["title"]
        merged_map[key] = {"doc": doc, "score": 1.0 / (k + rank + 1)}

    for rank, doc in enumerate(list_b):
        key = doc["title"]
        rrf_score = 1.0 / (k + rank + 1)
        if key in merged_map:
            merged_map[key]["score"] += rrf_score  # 两路都命中 → 加分
        else:
            merged_map[key] = {"doc": doc, "score": rrf_score}

    # 按 RRF 综合分数降序排列
    sorted_items = sorted(merged_map.values(), key=lambda x: x["score"], reverse=True)

    # 返回文档列表（带 RRF 分数），交给 Reranker 精排
    result = []
    for item in sorted_items:
        doc = item["doc"].copy()
        doc["score"] = round(item["score"], 4)
        result.append(doc)

    return result  # 返回所有融合结果，Reranker 会从中精排
