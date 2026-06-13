"""测试文档加载器"""
import pytest
from pathlib import Path
import tempfile
import os

# 确保项目路径在 sys.path 中
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.document_loader import load_documents, _read_file


class TestDocumentLoader:
    """测试从目录加载文档"""

    def test_read_single_file(self):
        """读取单个 .md 文件应返回 title 和 content"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# 测试标题\n\n这是测试内容。")
            f.flush()
            result = _read_file(Path(f.name))

        os.unlink(f.name)
        assert result["title"] != ""
        assert "测试内容" in result["content"]

    def test_load_documents_from_dir(self):
        """从目录加载应发现所有 .md 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建两个测试文件
            Path(tmpdir, "doc1.md").write_text("# 文档1\n内容1", encoding="utf-8")
            Path(tmpdir, "doc2.md").write_text("# 文档2\n内容2", encoding="utf-8")
            Path(tmpdir, "readme.txt").write_text("txt文件", encoding="utf-8")

            docs = load_documents(str(tmpdir))
            assert len(docs) == 3  # 2 md + 1 txt

    def test_empty_directory(self):
        """空目录应返回空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents(str(tmpdir))
            assert docs == []

    def test_skip_non_text_files(self):
        """非 .md/.txt 文件应被忽略"""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "image.png").write_text("not text", encoding="utf-8")
            docs = load_documents(str(tmpdir))
            assert len(docs) == 0

    def test_utf8_encoding(self):
        """UTF-8 中文文件应正确读取"""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "中文文档.md").write_text("# 数据库配置\n连接池最大连接数默认为20。", encoding="utf-8")
            docs = load_documents(str(tmpdir))
            assert len(docs) == 1
            assert "连接池" in docs[0]["content"]
            assert "中文文档" in docs[0]["title"]  # title = 文件名去后缀
