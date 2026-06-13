"""
SmartKB 聊天前端 —— Chainlit 驱动的对话界面
启动: chainlit run frontend/app.py -w --port 8002
"""
import chainlit as cl
import json
import urllib.request

API_BASE = "http://127.0.0.1:8001"


@cl.on_chat_start
async def start():
    await cl.Message(
        content="# SmartKB 企业级知识库问答系统\n\n"
                "当前知识库：MySQL连接池配置 / 用户密码重置流程 / API限流策略 / 系统运维手册\n\n"
                "**直接输入问题开始对话：**"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    question = message.content.strip()
    if not question:
        return

    # 自动拿 token
    try:
        login_data = json.dumps({"username": "user", "password": "smartkb"}).encode()
        login_req = urllib.request.Request(
            f"{API_BASE}/login",
            data=login_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(login_req, timeout=10) as resp:
            token = json.loads(resp.read())["access_token"]
    except Exception:
        token = ""

    # 调 /chat
    chat_data = json.dumps({"question": question, "session_id": ""}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(f"{API_BASE}/chat", data=chat_data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # 拼接来源
        elements = []
        if sources:
            refs = "\n\n---\n**📎 参考来源：**\n"
            for i, s in enumerate(sources[:3]):
                title = s.get("title", "未知")
                score = s.get("score", 0)
                snippet = s.get("snippet", "")[:300]
                refs += f"\n> [{i+1}] **{title}** (相关度: {score:.2f})"
                elements.append(
                    cl.Text(name=f"来源{i+1}: {title}", content=snippet, display="side")
                )
            answer += refs

        await cl.Message(content=answer, elements=elements).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ 请求失败：{e}").send()
