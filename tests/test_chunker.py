"""测试语义分块器"""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chunker import chunk_document, _split_paragraphs, _merge_paragraphs


class TestChunker:
    """测试文档分块逻辑"""

    def test_short_document_single_chunk(self):
        """短文档（< CHUNK_SIZE）应保持为 1 块"""
        doc = {"title": "短文", "content": "这是一篇很短的文档，只有几十个字。"}
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["title"] == "短文"
        assert chunks[0]["chunk_index"] == 0

    def test_long_document_multiple_chunks(self):
        """长文档应被切分多块"""
        # 构造一篇超过 500 字的文档
        paragraphs = [f"第{i}段: " + "测试内容" * 40 for i in range(5)]
        content = "\n\n".join(paragraphs)
        doc = {"title": "长文", "content": content}
        chunks = chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        """每块应保留源文档标题和块序号"""
        doc = {"title": "测试文档", "content": "正文" * 300}
        chunks = chunk_document(doc)
        for i, c in enumerate(chunks):
            assert c["title"] == "测试文档"
            assert c["chunk_index"] == i

    def test_paragraph_split_by_empty_lines(self):
        """按空行切段落"""
        text = "段落1\n\n段落2\n\n段落3"
        paras = _split_paragraphs(text)
        assert len(paras) == 3
        assert paras[0] == "段落1"

    def test_overlap_between_chunks(self):
        """相邻块之间应有重叠"""
        para = "ABCDEFGHIJ" * 30  # 300 字符，刚好成块
        text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
        doc = {"title": "overlap测试", "content": text}
        chunks = chunk_document(doc)
        if len(chunks) >= 2:
            # 第 2 块开头应以 "..."  开头（重叠标记）
            has_overlap = chunks[1]["content"].startswith("...") or len(chunks) == 1
            assert has_overlap or len(chunks) == 1

    def test_empty_document(self):
        """空文档应返回空列表"""
        doc = {"title": "空", "content": ""}
        chunks = chunk_document(doc)
        assert chunks == []
