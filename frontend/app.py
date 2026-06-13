"""
SmartKB 聊天前端 —— Chainlit 驱动的对话界面
启动: chainlit run frontend/app.py -w
面试能说: 用 Chainlit 快速搭建 LLM 应用原型界面，专注后端核心逻辑
"""
import chainlit as cl
import json
import urllib.request

# 后端 API 地址（Step 7 跑着的服务）
API_URL = "http://127.0.0.1:8001/chat"


@cl.on_chat_start
async def start():
    """用户打开聊天窗口时触发"""
    await cl.Message(
        content="👋 你好！我是 SmartKB 知识库助手。\n\n"
                "当前知识库包含：\n"
                "- 数据库连接池配置指南\n"
                "- 用户密码重置流程\n"
                "- API限流策略说明\n"
                "- 系统运维手册\n\n"
                "请直接输入你的问题。"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """用户发送消息时触发：调用后端 /chat 接口，返回答案+来源"""
    question = message.content

    # 调后端 API
    data = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # 拼接来源引用
        if sources:
            refs = "\n\n---\n**📎 参考来源：**\n"
            for i, s in enumerate(sources[:3]):
                title = s.get("title", "未知")
                score = s.get("score", 0)
                refs += f"\n> [{i+1}] **{title}** (相关度: {score:.2f})"

            full_answer = answer + refs
        else:
            full_answer = answer

        await cl.Message(content=full_answer).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ 请求失败：{e}\n请确认后端服务已启动（http://127.0.0.1:8001）").send()
