"""
Function Calling / Tool Use —— DeepSeek 原生工具调用
面试核心：不是硬编码调用链，而是定义 tool schema，LLM 自主判断调哪个工具、传什么参数。

DeepSeek Function Calling 兼容 OpenAI 的函数调用协议。
"""
import json
from openai import OpenAI
from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 1. 工具定义（Tool Schema）—— 面试重点
# ============================================================
# 每个工具就是一个函数签名 + 描述。LLM 根据描述和参数定义判断何时调用。
# 面试说法："我设计 tool schema 的原则——描述精确到能排除歧义，
#           参数带 enum 约束减少 LLM 传错参数的概率。"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在知识库中搜索相关文档。当用户问技术问题、查配置、找说明时使用。不要用于闲聊或天气等无关问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题。应该从用户问题中提取核心概念，去掉语气词。例如用户问'那个数据库连接池最大是多少来看' → 提取为'数据库连接池 最大连接数'",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["all", "database", "api", "user", "deployment"],
                        "description": "文档分类过滤。不确定时用 all。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_detail",
            "description": "获取某篇文档的完整内容。当用户想看完整文档、或者 search_documents 返回的摘要不够时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_title": {
                        "type": "string",
                        "description": "文档标题（从 search_documents 返回的 title 字段获取）",
                    },
                },
                "required": ["doc_title"],
            },
        },
    },
]


# ============================================================
# 2. 工具实现（Tool Implementation）
# ============================================================
class ToolExecutor:
    """
    工具执行器 —— 接收 LLM 返回的 tool_call，执行并返回结果。
    面试说法："每个工具是一个纯函数，输入参数、输出字符串。
              新增工具只需两个步骤：1) 在 TOOLS 列表加 schema
              2) 在这个类加一个方法。调度逻辑不用改。"
    """

    def __init__(self, store=None, sparse_index=None, embedder=None, documents=None):
        self._store = store
        self._sparse_index = sparse_index
        self._embedder = embedder
        self._documents = documents or []

    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用，返回字符串结果给 LLM"""
        if tool_name == "search_documents":
            return self._search_documents(arguments.get("query", ""), arguments.get("category", "all"))
        elif tool_name == "get_document_detail":
            return self._get_document_detail(arguments.get("doc_title", ""))
        else:
            return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

    def _search_documents(self, query: str, category: str = "all") -> str:
        """搜索文档（简化版——直接调现有检索，不做分类过滤）"""
        if not query:
            return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

        # 如果有 embedder 和 store，走真正的混合检索
        if self._embedder and self._store and self._sparse_index:
            try:
                from src.rag.retriever import retrieve
                from src.rag.reranker import Reranker
                query_dense = self._embedder.encode([query])[0]
                query_sparse = self._embedder.encode_sparse([query])[0]
                hits = retrieve(
                    query=query,
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    store=self._store,
                    sparse_index=self._sparse_index,
                    reranker=Reranker(),
                    top_k=3,
                )
                results = [
                    {"title": h["title"], "content": h["content"][:300], "score": h["score"]}
                    for h in hits
                ]
                return json.dumps({"results": results}, ensure_ascii=False)
            except Exception as e:
                logger.error("tool_search_failed", extra={"query": query, "error": str(e)})
                return json.dumps({"error": f"搜索失败: {e}"}, ensure_ascii=False)

        # Fallback：纯文本关键词匹配
        results = []
        for doc in self._documents:
            if query.lower() in doc.get("content", "").lower():
                results.append({
                    "title": doc.get("title", ""),
                    "content": doc.get("content", "")[:300],
                    "score": 0.5,
                })
        return json.dumps({"results": results[:3]}, ensure_ascii=False)

    def _get_document_detail(self, doc_title: str) -> str:
        """获取文档完整内容"""
        if not doc_title:
            return json.dumps({"error": "doc_title 不能为空"}, ensure_ascii=False)
        for doc in self._documents:
            if doc_title in doc.get("title", ""):
                return json.dumps({
                    "title": doc["title"],
                    "content": doc["content"],
                }, ensure_ascii=False)
        return json.dumps({"error": f"未找到文档: {doc_title}"}, ensure_ascii=False)


# ============================================================
# 3. Function Calling 核心循环 —— 面试必问
# ============================================================
def chat_with_tools(
    user_question: str,
    tool_executor: ToolExecutor,
    max_turns: int = 3,
) -> dict:
    """
    带工具调用的对话循环。
    LLM 先推理 → 需要工具 → 调用 → 拿到结果 → 再推理 → 最终回答。

    面试说法："实现的是标准的 ReAct 循环：Reason → Act → Observe → Reason → Answer。
            max_turns=3 防止无限循环。每轮 LLM 可能调 0-1 个工具。
            不调到直接回答。"
    """
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    messages = [{"role": "user", "content": user_question}]

    logger.info("tool_chat_start", extra={"question": user_question[:80]})

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )

        msg = response.choices[0].message

        # 情况 1：模型决定不调工具，直接回答
        if not msg.tool_calls:
            logger.info("tool_chat_no_tool_needed", extra={"answer": msg.content[:80] if msg.content else "empty"})
            return {
                "answer": msg.content or "",
                "tool_calls_made": turn,
                "reasoning": "模型判断不需要调工具，直接回答",
            }

        # 情况 2：模型要调工具
        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        logger.info("tool_chat_calling", extra={
            "turn": turn + 1,
            "tool": tool_name,
            "tool_args": str(arguments)[:100],
        })

        # 执行工具
        result = tool_executor.execute(tool_name, arguments)

        # 把工具调用和结果加到消息历史
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_call.function.arguments},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # 最后一轮：强制生成最终回答
    final_response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.1,
    )
    return {
        "answer": final_response.choices[0].message.content or "",
        "tool_calls_made": max_turns,
        "reasoning": f"达到最大轮次 {max_turns}，强制生成回答",
    }
