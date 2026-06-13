# SmartKB - Step 1: 最简RAG (Level 0 - 理解原理)
# 目标：跑通"文档→嵌入→检索→回答"完整链路
# 硬编码3篇文档，内存向量检索，DeepSeek生成回答

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# 1. 准备你的"知识库" —— 3篇模拟文档
# ============================================================
documents = [
    {
        "title": "数据库连接池配置",
        "content": "连接池最大连接数默认值为20，最小为5。当连接数超过最大值时，新请求将进入等待队列，超时时间为30秒。可以在 config.yaml 中的 database.pool 节点修改此参数。",
    },
    {
        "title": "用户密码重置流程",
        "content": "管理员登录后台管理系统，进入'用户管理'页面，搜索目标用户，点击'重置密码'按钮。系统将自动向该用户的注册邮箱发送一个有效期为15分钟的重置链接。新密码必须包含至少8个字符，包括大小写字母和数字。",
    },
    {
        "title": "API限流策略说明",
        "content": "系统对API请求实施基于令牌桶算法的限流。默认配置为每秒100个请求，突发峰值允许到200个。超过限流阈值时，API返回429状态码，响应体中包含Retry-After头指示重试时间。限流配置可在 config.yaml 的 rate_limit 节点修改。",
    },
]

# ============================================================
# 2. 加载嵌入模型（通过 ModelScope，国内直连）
# ============================================================
print("正在从 ModelScope 下载 BGE-M3 嵌入模型（约2GB，仅首次）...")
from modelscope import snapshot_download

model_dir = snapshot_download("BAAI/bge-m3", cache_dir="./models")
embed_model = SentenceTransformer(model_dir)

# 把所有文档内容转成向量
doc_texts = [doc["content"] for doc in documents]
doc_embeddings = embed_model.encode(doc_texts)  # shape: (3, 1024)
print(f"文档数量: {len(documents)}, 向量维度: {doc_embeddings.shape[1]}")


# ============================================================
# 3. 检索函数 —— 用余弦相似度找最相关的文档
# ============================================================
def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """输入用户问题，返回最相关的文档片段"""
    # 把用户问题也转成向量
    query_embedding = embed_model.encode([query])[0]

    # 计算问题向量与每个文档向量的余弦相似度
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = np.dot(query_embedding, doc_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
        )
        similarities.append((i, sim))

    # 按相似度从高到低排序，取 top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, sim in similarities[:top_k]:
        results.append(
            {"title": documents[i]["title"], "content": documents[i]["content"], "score": round(float(sim), 4)}
        )
    return results


# ============================================================
# 4. 生成回答 —— 检索结果拼到 Prompt，交给 LLM
# ============================================================
def generate_answer(query: str, retrieved_docs: list[dict]) -> str:
    """把检索到的文档和用户问题拼成 Prompt，让 LLM 生成回答"""
    # 拼接上下文
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"[来源{i+1}] {doc['title']}\n{doc['content']}")
    context = "\n\n".join(context_parts)

    # 构造 Prompt
    prompt = f"""你是一个技术文档助手。请严格根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请直接说"根据现有文档无法回答"。

参考资料：
{context}

用户问题：{query}

请回答："""

    # 调用 DeepSeek API（兼容 OpenAI SDK）
    client = OpenAI(
        api_key="sk-41c4b9b9c9bf4c2e885a6a146e32943b",
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 低温度 = 减少随机性 = 回答更忠实于文档
    )
    return response.choices[0].message.content


# ============================================================
# 5. 跑起来！
# ============================================================
if __name__ == "__main__":
    # 测试问题1：精确匹配
    query1 = "数据库连接池默认大小是多少？"
    print(f"\n{'='*60}")
    print(f"用户问题: {query1}")
    print(f"{'='*60}")

    docs = retrieve(query1, top_k=2)
    print("\n检索结果:")
    for d in docs:
        print(f"  [{d['score']}] {d['title']}: {d['content'][:60]}...")

    answer = generate_answer(query1, docs)
    print(f"\nAI回答: {answer}")

    # 测试问题2：跨文档
    query2 = "配置文件里能改哪些参数？"
    print(f"\n{'='*60}")
    print(f"用户问题: {query2}")
    print(f"{'='*60}")

    docs = retrieve(query2, top_k=2)
    print("\n检索结果:")
    for d in docs:
        print(f"  [{d['score']}] {d['title']}: {d['content'][:60]}...")

    answer = generate_answer(query2, docs)
    print(f"\nAI回答: {answer}")
