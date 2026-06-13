"""
生成器 —— 拼 Prompt + 调 DeepSeek API
面试要点：Prompt 模板设计是 RAG 核心工程实践，temperature 和 grounding 指令是关键
"""
from openai import OpenAI
from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def generate_answer(query: str, retrieved_docs: list[dict]) -> str:
    """检索结果 + 用户问题 → LLM 生成回答"""
    # 拼接上下文
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"[来源{i+1}] {doc['title']}\n{doc['content']}")
    context = "\n\n".join(context_parts)

    # 构造 Prompt（这是面试会问的——为什么这么设计）
    prompt = f"""你是一个技术文档助手。请严格根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请直接说"根据现有文档无法回答"。

参考资料：
{context}

用户问题：{query}

请回答："""

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content
