"""测试 RRF 融合算法"""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import _rrf_fusion


class TestRRFFusion:
    """测试倒数排名融合"""

    def make_doc(self, title, score=0.5):
        return {"title": title, "content": "test content", "score": score}

    def test_single_list(self):
        """只有一路召回时，结果等于该路排序"""
        a = [self.make_doc("A", 0.9), self.make_doc("B", 0.7), self.make_doc("C", 0.5)]
        result = _rrf_fusion(a, [], k=60)
        assert len(result) == 3
        assert result[0]["title"].startswith("A")

    def test_both_hit_boosts_score(self):
        """两路都命中的文档分数应高于单路命中"""
        a = [self.make_doc("shared"), self.make_doc("only_a")]
        b = [self.make_doc("shared"), self.make_doc("only_b")]
        result = _rrf_fusion(a, b, k=60)
        # shared 在两路都排第一 → 分数最高
        assert result[0]["title"].startswith("shared")
        # shared 分数 > only_a 或 only_b 的分数
        shared_score = result[0]["score"]
        assert shared_score > result[1]["score"]

    def test_deduplication(self):
        """两路召回相同文档不应重复"""
        a = [self.make_doc("X"), self.make_doc("Y")]
        b = [self.make_doc("X"), self.make_doc("Z")]
        result = _rrf_fusion(a, b, k=60)
        assert len(result) == 3  # X, Y, Z 各一条

    def test_score_range(self):
        """RRF 分数应在合理范围内"""
        a = [self.make_doc(f"doc_{i}") for i in range(10)]
        b = [self.make_doc(f"doc_{i}") for i in range(5)]  # 重叠前 5
        result = _rrf_fusion(a, b, k=60)
        assert len(result) == 10
        for r in result:
            assert 0 < r["score"] < 0.1  # RRF 公式：1/(k+rank)

    def test_ranking_consistency(self):
        """同一文档在两路都排前面的应排在最前"""
        a = [self.make_doc("C"), self.make_doc("A"), self.make_doc("B")]
        b = [self.make_doc("B"), self.make_doc("A"), self.make_doc("C")]
        result = _rrf_fusion(a, b, k=60)
        # A: 1/(60+2) + 1/(60+2) = 0.01587 + 0.01587 = 0.03174
        # B: 1/(60+3) + 1/(60+1) = 0.01563 + 0.01639 = 0.03202
        # C: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01563 = 0.03202
        assert result[0]["title"].startswith("B") or result[0]["title"].startswith("C")
