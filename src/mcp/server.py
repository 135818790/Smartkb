"""
MCP Server —— 把 SmartKB 检索和问答封装为标准 MCP 工具
面试核心：MCP 协议 = AI 应用的 USB 接口。一次开发，多个客户端调用。

启动: python src/mcp/server.py
客户端 (Cursor/Claude Desktop) 配置:
  {
    "mcpServers": {
      "smartkb": {
        "command": "python",
        "args": ["src/mcp/server.py"],
        "cwd": "/path/to/smartkb"
      }
    }
  }

面试说法："MCP 有三个核心概念——
  Server: 我暴露的工具服务
  Client: Cursor/Claude Desktop 等 AI 应用
  Transport: stdio(本地进程通信) 或 SSE(远程 HTTP)

  我实现的是 stdio transport——Client 启动我的 Server 进程，
  通过 stdin/stdout 交换 JSON-RPC 消息。不需要网络端口，
  安全且适合本地工具。"
"""
import json
import sys
from pathlib import Path

# 确保项目根在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.core.config import DEEPSEEK_API_KEY

logger = get_logger(__name__)


# ============================================================
# 1. 工具实现（纯函数，每句话面试都能解释）
# ============================================================
def _search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """搜索 SmartKB 知识库"""
    if not query:
        return {"error": "query 不能为空"}
    try:
        from src.rag.retriever import retrieve
        from src.rag.embedder import Embedder
        from src.rag.vector_store import VectorStore
        from src.rag.sparse_index import SparseIndex
        from src.rag.reranker import Reranker

        embedder = Embedder()
        store = VectorStore()
        sparse_index = SparseIndex()

        # 懒加载——首次请求时检查并导入文档
        if store.is_empty:
            from src.rag.document_loader import load_documents
            from src.rag.chunker import chunk_document
            docs = load_documents()
            chunks = []
            for d in docs:
                chunks.extend(chunk_document(d))
            texts = [c["content"] for c in chunks]
            dense, sparse = embedder.encode_both(texts)
            store.add_documents(chunks, dense)
            sparse_index.add(chunks, sparse)

        query_dense = embedder.encode([query])[0]
        query_sparse = embedder.encode_sparse([query])[0]
        hits = retrieve(query, query_dense, query_sparse, store, sparse_index, Reranker(), top_k)

        return {
            "results": [
                {"title": h["title"], "snippet": h["content"][:200], "score": h["score"]}
                for h in hits
            ]
        }
    except Exception as e:
        return {"error": str(e)}


def _ask_knowledge_base(question: str) -> dict:
    """用自然语言提问，返回 AI 生成的回答"""
    if not question:
        return {"error": "question 不能为空"}
    if not DEEPSEEK_API_KEY:
        return {"error": "DeepSeek API Key 未配置"}
    try:
        from openai import OpenAI
        # 先检索
        search_result = _search_knowledge_base(question, top_k=3)
        results = search_result.get("results", [])

        # 拼 Prompt 生成
        contexts = "\n\n".join(r["snippet"] for r in results)
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"参考资料：\n{contexts}\n\n问题：{question}\n请严格根据参考资料回答。",
            }],
            temperature=0.1,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": [r["title"] for r in results],
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 2. MCP Server（stdio transport —— 面试重点）
# ============================================================
def run_mcp_server():
    """
    MCP stdio server 主循环。
    通过 stdin 读 JSON-RPC 请求，stdout 写 JSON-RPC 响应。
    面试说法："MCP 的 stdio transport 是最简单且最安全的方式——
              不需要网络端口，Client 启动进程，通过标准输入输出通信。""
    """
    logger.info("mcp_server_starting")

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        req_id = request.get("id")

        # --- initialize: 握手 ---
        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "smartkb-mcp",
                        "version": "1.0.0",
                    },
                    "capabilities": {
                        "tools": {},
                    },
                },
            }

        # --- tools/list: 告诉客户端有哪些工具 ---
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "search_smartkb",
                            "description": "搜索 SmartKB 知识库，返回相关文档片段和相似度分数。当需要查找技术文档、配置说明、运维手册时使用。",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "搜索关键词或问题",
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "返回结果数（默认 3）",
                                        "default": 3,
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "ask_smartkb",
                            "description": "用自然语言向 SmartKB 提问，返回 AI 生成的回答和引用来源。适合需要完整回答而非片段检索的场景。",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "自然语言问题",
                                    },
                                },
                                "required": ["question"],
                            },
                        },
                    ]
                },
            }

        # --- tools/call: 执行工具 ---
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name == "search_smartkb":
                result = _search_knowledge_base(
                    arguments.get("query", ""),
                    arguments.get("top_k", 3),
                )
            elif tool_name == "ask_smartkb":
                result = _ask_knowledge_base(arguments.get("question", ""))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                },
            }

        # --- 其他方法 ---
        elif method == "notifications/initialized":
            continue  # 不需要回复

        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_server()
