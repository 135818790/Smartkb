"""
LangGraph Agentic RAG —— 有状态多步推理
面试核心：不用 LangChain 的预置链，用 LangGraph 自己搭状态机。
          体现的是"我理解 Agent 的原理"，不是"我会调 API"。

图结构:
  START → router → retrieve → generate → verify → END
                    ↑                        │
                    └────── retry ────────────┘ (verify 不通过时)
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from src.core.config import TOP_K


# ============================================================
# 1. 状态定义 —— 所有节点共享的数据结构
# ============================================================
class AgentState(TypedDict):
    question: str              # 用户原始问题
    query_dense: list          # 问题的稠密向量
    query_sparse: dict         # 问题的稀疏向量
    question_type: str         # simple / complex / out_of_scope
    refined_query: str         # 重试时改写后的问题
    history: list[str]         # E3: 多轮对话历史
    hits: list[dict]           # 检索到的文档块
    answer: str                # 生成的回答
    verification: str          # pass / fail / uncertain
    retry_count: int           # 当前重试次数


# ============================================================
# 2. 节点函数 —— 每个节点是纯函数，输入 State，输出 State
# ============================================================

def router_node(state: AgentState, *, llm_client) -> AgentState:
    """
    路由器：判断问题类型。
    简单问题走标准 RAG，复杂问题走多步推理，无关问题拒绝。
    面试能说：用一个轻量 LLM 调用做意图分类，避免复杂问题硬答。
    """
    question = state["question"]

    prompt = f"""分析以下用户问题，判断类型。只回复一个词。

问题：{question}

类型：
- simple: 简单事实查询，直接检索即可回答
- complex: 需要多步推理、对比分析或跨文档整合
- out_of_scope: 与技术文档完全无关的问题

类型："""

    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    result = resp.choices[0].message.content.strip().lower()

    if "complex" in result:
        state["question_type"] = "complex"
    elif "out_of_scope" in result or "无关" in result:
        state["question_type"] = "out_of_scope"
    else:
        state["question_type"] = "simple"

    state["retry_count"] = 0
    return state


def retrieve_node(state: AgentState, *, embedder, store, sparse_index, reranker) -> AgentState:
    """
    检索节点：执行混合检索。
    Step 5 的 retrieve() 原封不动搬进来。
    """
    from src.rag.retriever import retrieve
    from src.rag.reranker import Reranker

    # 用 refined_query（如果有的话），否则用原始问题
    search_query = state.get("refined_query") or state["question"]

    # 重试时需要重新编码改写后的问题
    if state.get("refined_query"):
        state["query_dense"] = embedder.encode([search_query])[0]
        state["query_sparse"] = embedder.encode_sparse([search_query])[0]

    hits = retrieve(
        query=search_query,
        query_dense=state["query_dense"],
        query_sparse=state["query_sparse"],
        store=store,
        sparse_index=sparse_index,
        reranker=reranker,
        top_k=TOP_K,
    )
    state["hits"] = hits
    return state


def generate_node(state: AgentState, *, llm_client) -> AgentState:
    """
    生成节点：检索结果 + 问题 → LLM 生成回答。
    面试能说：Prompt 设计是 RAG 核心工程实践，这里加了两层约束。
    """
    # 拼接上下文
    context_parts = []
    for i, doc in enumerate(state["hits"]):
        context_parts.append(f"[来源{i+1}] {doc['title']}\n{doc['content']}")
    context = "\n\n".join(context_parts)

    # Prompt：两层约束 —— grounding + 诚实
    prompt = f"""你是一个技术文档助手。请严格根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请直接说"根据现有文档无法回答"。
如果信息不完整，指出缺少什么信息。
注意：如果用户的问题是对上一轮对话的追问，请结合对话历史给出连贯的回答。

对话历史：
{chr(10).join(state.get('history', []))}

参考资料：
{context}

用户问题：{state['question']}

请回答："""

    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    state["answer"] = resp.choices[0].message.content
    return state


def verify_node(state: AgentState, *, llm_client, embedder, store, sparse_index, reranker) -> AgentState:
    """
    验证节点（Self-RAG 核心）：
    LLM 自己检查回答是否有文档依据。不通过 → 改写问题重试。
    面试能说：Self-RAG = 生成后自检，幻觉检测不依赖外部模型。
    """
    # 拼接回答和来源，让 LLM 逐句核对
    sources_text = "\n".join(
        f"[{i+1}] {h['title']}: {h['content'][:300]}"
        for i, h in enumerate(state["hits"])
    )

    check_prompt = f"""你是回答质量检查员。判断以下回答是否忠实于参考资料。

参考资料：
{sources_text}

回答：
{state['answer']}

请判断：
- 回答的每句话是否都能在参考资料中找到依据？
- 如果找不到依据，回答是否诚实说明了？

只回复：pass（完全有依据）、fail（有明显编造）、uncertain（部分依据不足）

判定："""

    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": check_prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    result = resp.choices[0].message.content.strip().lower()

    if "pass" in result:
        state["verification"] = "pass"
    elif "fail" in result:
        state["verification"] = "fail"
        state["retry_count"] += 1

        # 改写问题：让 LLM 生成更具体的关键词查询
        if state["retry_count"] <= 1:  # 最多重试 1 次
            refine_prompt = f"""以下问题的回答未能从文档中找到充分依据。
请将问题改写为更具体的关键词搜索查询，便于从技术文档中检索。

原始问题：{state['question']}

只输出改写后的查询："""
            resp2 = llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            state["refined_query"] = resp2.choices[0].message.content.strip()
    else:
        state["verification"] = "uncertain"

    return state


# ============================================================
# 3. 条件路由函数 —— 决定下一步去哪
# ============================================================

def route_after_router(state: AgentState) -> Literal["retrieve", "reject"]:
    """路由后：简单/复杂→检索，无关→拒绝"""
    if state["question_type"] == "out_of_scope":
        return "reject"
    return "retrieve"


def route_after_verify(state: AgentState) -> Literal["retrieve", "end"]:
    """验证后：通过/不确定→结束，失败且未超限→重试"""
    if state["verification"] == "fail" and state["retry_count"] <= 1:
        return "retrieve"
    return "end"


# ============================================================
# 4. 构建图 —— 节点 + 边 = 状态机
# ============================================================

def build_graph(llm_client, embedder, store, sparse_index, reranker):
    """
    构建 LangGraph 状态图。
    面试能说：用 StateGraph 而非 Chain，因为 Agent 需要条件分支和循环。
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("router", lambda s: router_node(s, llm_client=llm_client))
    graph.add_node(
        "retrieve",
        lambda s: retrieve_node(
            s, embedder=embedder, store=store,
            sparse_index=sparse_index, reranker=reranker,
        ),
    )
    graph.add_node("generate", lambda s: generate_node(s, llm_client=llm_client))
    graph.add_node(
        "verify",
        lambda s: verify_node(
            s, llm_client=llm_client, embedder=embedder,
            store=store, sparse_index=sparse_index, reranker=reranker,
        ),
    )
    graph.add_node("reject", lambda s: _reject_node(s))

    # 注册边
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router, {
        "retrieve": "retrieve",
        "reject": "reject",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {
        "retrieve": "retrieve",
        "end": END,
    })
    graph.add_edge("reject", END)

    return graph.compile()


def _reject_node(state: AgentState) -> AgentState:
    """拒绝无关问题"""
    state["answer"] = "抱歉，您的问题与知识库内容无关。请提出与技术文档相关的问题。"
    state["verification"] = "pass"
    return state
