"""
文档分块器 —— 语义分块，按段落切分，小块合并
"""
import re
from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.core.exceptions import ChunkingError


def chunk_document(document: dict) -> list[dict]:
    """
    将一篇文档切成多个小块。每块保留源文档信息。
    """
    title = document["title"]
    text = document.get("content", "")

    if not text:
        return []

    try:
        paragraphs = _split_paragraphs(text)
        chunks = _merge_paragraphs(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP)

        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                "title": title,
                "chunk_index": i,
                "content": chunk_text,
            })
        return result
    except Exception as e:
        raise ChunkingError(f"文档分块失败: {title}", details={"title": title, "text_len": len(text)}, cause=e)


def _split_paragraphs(text: str) -> list[str]:
    """按空行切段落，过滤掉纯标题行"""
    # 先按空行切
    raw = re.split(r"\n\s*\n", text)
    # 过滤空段落
    return [p.strip() for p in raw if p.strip()]


def _merge_paragraphs(paragraphs: list[str], target_size: int, overlap: int) -> list[str]:
    """
    合并短段落，直到接近 target_size。
    相邻块之间保留 overlap 个字符的重叠。
    """
    chunks = []
    current = ""
    for para in paragraphs:
        # 如果当前块 + 新段落没超过目标，直接拼上
        if len(current) + len(para) <= target_size:
            current += para + "\n\n"
        else:
            # 当前块够大了，保存它
            if current.strip():
                chunks.append(current.strip())

            # 新块开头：从前一块末尾取 overlap 个字符作为上下文
            if chunks and overlap > 0:
                prev = chunks[-1]
                # 从上一块的末尾取 overlap 个字符
                tail = prev[-overlap:] if len(prev) >= overlap else prev
                current = f"...{tail}\n\n{para}\n\n"
            else:
                current = para + "\n\n"

    # 最后一个块
    if current.strip():
        chunks.append(current.strip())

    return chunks
