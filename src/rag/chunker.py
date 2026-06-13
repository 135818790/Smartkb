"""
文档分块器 —— 语义分块，按段落切分，小块合并
面试要点：分块策略直接影响检索质量。太大→噪声多，太小→语义不完整。
"""
import re
from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(document: dict) -> list[dict]:
    """
    将一篇文档切成多个小块。每块保留源文档信息。
    输入: {"title": "xxx", "content": "正文..."}
    输出: [{"title": "xxx", "chunk_index": 0, "content": "第0块..."}, ...]
    """
    title = document["title"]
    text = document["content"]

    # 第一步：按段落切分（空行 = 段落边界）
    paragraphs = _split_paragraphs(text)

    # 第二步：合并短段落，直到接近目标大小
    chunks = _merge_paragraphs(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP)

    # 第三步：每块带上源文档信息
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "title": title,
            "chunk_index": i,
            "content": chunk_text,
        })
    return result


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
