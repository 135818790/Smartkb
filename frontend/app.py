"""
SmartKB 聊天前端 —— Chainlit 驱动的对话界面
启动: chainlit run frontend/app.py -w
支持: 文字问答 + 文件上传 + JWT 登录
"""
import chainlit as cl
import json
import urllib.request

API_BASE = "http://127.0.0.1:8001"
CHAT_URL = f"{API_BASE}/chat"
UPLOAD_URL = f"{API_BASE}/documents/upload"
STATUS_URL = f"{API_BASE}/documents/status"


@cl.on_chat_start
async def start():
    """用户打开聊天窗口时触发——先登录获取 JWT，保存到用户 session"""
    await cl.Message(
        content="🔐 正在验证身份...\n\n输入用户名和密码（开发环境接受任意凭据）："
    ).send()

    # 等待用户输入凭据（格式：用户名 密码）
    creds_msg = await cl.AskUserMessage(content="请输入 用户名 密码（空格分隔）", timeout=60).send()
    if creds_msg:
        parts = creds_msg["output"].strip().split()
        if len(parts) >= 2:
            username, password = parts[0], parts[1]
            try:
                token = _api_post("/login", {"username": username, "password": password})
                cl.user_session.set("token", token["access_token"])
                await cl.Message(
                    content=f"✅ 登录成功，欢迎 {username}！\n\n"
                            "📁 支持上传 .md / .txt 文件到知识库\n"
                            "💬 直接输入问题开始问答\n\n"
                            "**输入问题开始对话：**"
                ).send()
                return
            except Exception as e:
                pass

    # 登录失败也用默认 token（开发友好）
    cl.user_session.set("token", "")
    await cl.Message(
        content="⚠️ 未登录，/chat 接口可能受限。\n\n"
                "**输入问题开始对话：**"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """用户发送消息或上传文件"""
    token = cl.user_session.get("token", "")

    # 如果有文件附件 → 上传到知识库
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path"):
                await _handle_upload(element, token)
        return

    # 文字消息 → 问答
    question = message.content.strip()
    if not question:
        return

    data = json.dumps({"question": question}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(CHAT_URL, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        answer = result.get("answer", "")
        sources = result.get("sources", [])

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

    except urllib.error.HTTPError as e:
        if e.code == 401:
            await cl.Message(content="🔐 请先登录：发送 '登录 用户名 密码'").send()
        else:
            await cl.Message(content=f"⚠️ 请求失败 [{e.code}]").send()
    except Exception as e:
        await cl.Message(content=f"⚠️ 请求失败：{e}").send()


async def _handle_upload(element, token: str):
    """上传文件到知识库，轮询进度"""
    import os.path

    filename = os.path.basename(element.path)

    msg = cl.Message(content=f"📤 正在上传 **{filename}** ...")
    await msg.send()

    try:
        # 上传文件
        import http.client
        import mimetypes

        boundary = "----SmartKBUpload"
        body = _build_multipart(element.path, filename, boundary)

        req = urllib.request.Request(
            UPLOAD_URL,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {token}" if token else "",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            upload_result = json.loads(resp.read().decode("utf-8"))

        task_id = upload_result["task_id"]
        msg.content = f"📤 文件已上传，后台处理中...\n任务ID: `{task_id}`"
        await msg.update()

        # 轮询进度（最多等 60 秒）
        import time
        for _ in range(30):
            time.sleep(2)
            status_req = urllib.request.Request(f"{STATUS_URL}/{task_id}")
            with urllib.request.urlopen(status_req, timeout=10) as resp:
                status = json.loads(resp.read().decode("utf-8"))

            state = status.get("state", "")
            progress = status.get("progress", 0)

            if state == "SUCCESS":
                result = status.get("result", {})
                chunks = result.get("chunks", 0)
                msg.content = f"✅ **{filename}** 处理完成！共 {chunks} 个块，已加入知识库。\n\n现在你可以提问关于这个文档的问题了。"
                await msg.update()
                return

            if state == "FAILURE":
                msg.content = f"❌ **{filename}** 处理失败：{status.get('step', '未知错误')}"
                await msg.update()
                return

            # 更新进度
            if progress > 0:
                bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
                msg.content = f"📤 **{filename}** 处理中...\n{status.get('step', '处理中')} [{bar}] {progress}%"
                await msg.update()

    except Exception as e:
        msg.content = f"❌ 上传失败：{e}"
        await msg.update()


def _build_multipart(file_path: str, filename: str, boundary: str) -> bytes:
    """手动构建 multipart/form-data 请求体（避免依赖 requests 库）"""
    body = []
    body.append(f"--{boundary}".encode())
    body.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode())
    body.append(b"Content-Type: application/octet-stream")
    body.append(b"")
    with open(file_path, "rb") as f:
        body.append(f.read())
    body.append(f"--{boundary}--".encode())
    return b"\r\n".join(body)


def _api_post(path: str, data: dict) -> dict:
    """发 POST 请求到后端"""
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))
