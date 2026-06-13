"""
文档加载器 —— 从 data/documents/ 目录读取所有 .md 和 .txt 文件
"""
from pathlib import Path
from src.core.config import DOCUMENTS_DIR
from src.core.exceptions import DocumentLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_documents(docs_dir: str = DOCUMENTS_DIR) -> list[dict]:
    """
    扫描目录，读取所有 .md 和 .txt 文件
    返回: [{"title": "文件名", "content": "文件正文"}, ...]
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise DocumentLoadError(f"文档目录不存在: {docs_dir}")

    documents = []
    for filepath in docs_path.glob("*.md"):
        documents.append(_read_file(filepath))
    for filepath in docs_path.glob("*.txt"):
        documents.append(_read_file(filepath))

    if not documents:
        logger.warning("no_documents_found", extra={"dir": docs_dir})

    return documents


def _read_file(filepath: Path) -> dict:
    """读取单个文件，返回 {title, content}"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except (UnicodeDecodeError, OSError) as e:
        raise DocumentLoadError(f"文件读取失败: {filepath.name}", cause=e)

    return {
        "title": filepath.stem,
        "content": content,
    }
