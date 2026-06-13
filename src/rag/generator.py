"""
生成器 —— 拼 Prompt + 调 DeepSeek API（支持流式和非流式）
面试要点：Prompt 模板设计是 RAG 核心工程实践，streaming 是用户体验关键
"""
from typing import Generator
from openai import OpenAI
from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_prompt(query: str, retrieved_docs: list[dict], history: list[str] = None) -> str:
    """构造 RAG Prompt：历史对话 + 参考资料 + 用户问题"""
    # 历史对话上下文
    history_block = ""
    if history and len(history) > 0:
        # 取最近 6 轮
        recent = history[-6:]
        history_block = "对话历史：\n" + "\n".join(recent) + "\n\n"

    # 拼接检索到的文档
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"[来源{i+1}] {doc['title']}\n{doc['content']}")
    context = "\n\n".join(context_parts)

    return f"""你是一个技术文档助手。请严格根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请直接说"根据现有文档无法回答"。
注意：如果用户的问题是对上一轮问题的追问（如"那第二个呢"、"能详细说说吗"），
请结合对话历史和参考资料给出连贯的回答。

{history_block}参考资料：
{context}

用户问题：{query}

请回答："""


def generate_answer(query: str, retrieved_docs: list[dict], history: list[str] = None) -> str:
    """非流式：检索结果 + 用户问题 → LLM 生成回答（一次性返回）"""
    prompt = _build_prompt(query, retrieved_docs, history)
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


def generate_answer_stream(
    query: str, retrieved_docs: list[dict], history: list[str] = None
) -> Generator[str, None, None]:
    """
    流式：逐 token yield，配合 FastAPI StreamingResponse 实现 SSE。
    面试说法："用 Server-Sent Events 做流式输出，前端不用等完整结果，
              用户感知延迟从 5-10 秒降到 500ms 首字。"
    """
    prompt = _build_prompt(query, retrieved_docs, history)
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    logger.info("stream_start", extra={"query_len": len(query), "docs": len(retrieved_docs)})

    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        stream=True,
    )

    token_count = 0
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token_count += 1
            yield chunk.choices[0].delta.content

    logger.info("stream_end", extra={"tokens": token_count})
