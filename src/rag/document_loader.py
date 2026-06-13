"""
文档加载器 —— 从 data/documents/ 目录读取所有 .md 和 .txt 文件
面试要点：这是 RAG 系统的"数据入口"，生产环境要扩展支持 PDF、Word 等
"""
from pathlib import Path
from src.core.config import DOCUMENTS_DIR


def load_documents(docs_dir: str = DOCUMENTS_DIR) -> list[dict]:
    """
    扫描目录，读取所有 .md 和 .txt 文件
    返回: [{"title": "文件名", "content": "文件正文"}, ...]
    """
    docs_path = Path(docs_dir)
    documents = []

    for filepath in docs_path.glob("*.md"):
        documents.append(_read_file(filepath))
    for filepath in docs_path.glob("*.txt"):
        documents.append(_read_file(filepath))

    return documents


def _read_file(filepath: Path) -> dict:
    """读取单个文件，返回 {title, content}"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return {
        "title": filepath.stem,  # 文件名去后缀，如 "数据库连接池配置"
        "content": content,
    }
