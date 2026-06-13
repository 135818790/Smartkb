"""集成测试 —— 需要服务运行中"""
import pytest
from pathlib import Path
import sys
import urllib.request
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://127.0.0.1:8001"


def _post(path, data):
    """发送 POST 请求"""
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


@pytest.mark.integration
class TestHealthEndpoint:
    """健康检查接口"""

    def test_health_returns_ok(self):
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"


@pytest.mark.integration
class TestChatEndpoint:
    """聊天接口"""

    def test_simple_question_returns_answer(self):
        result = _post("/chat", {"question": "连接池最大连接数是多少？"})
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "session_id" in result

    def test_answer_contains_sources(self):
        result = _post("/chat", {"question": "API限流默认值？"})
        assert "sources" in result
        assert len(result["sources"]) > 0
        for s in result["sources"]:
            assert "title" in s
            assert "score" in s

    def test_out_of_scope_question(self):
        result = _post("/chat", {"question": "今天天气怎么样？"})
        # 不应崩溃，应返回有意义的内容
        assert "answer" in result

    def test_session_continuity(self):
        """多轮对话：第二次请求用同一个 session_id"""
        r1 = _post("/chat", {"question": "连接池最大连接数默认值？"})
        assert "session_id" in r1
        sid = r1["session_id"]
        r2 = _post("/chat", {"question": "那最小连接数呢？", "session_id": sid})
        assert "answer" in r2

    def test_empty_question(self):
        """空问题不应崩溃"""
        result = _post("/chat", {"question": ""})
        assert "answer" in result


@pytest.mark.integration
class TestStreamEndpoint:
    """流式接口"""

    def test_stream_returns_sse(self):
        import urllib.request as ur
        req = ur.Request(
            f"{BASE_URL}/chat/stream",
            data=json.dumps({"question": "连接池默认值？"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with ur.urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get("Content-Type", "")
            assert "text/event-stream" in content_type or "text/plain" in content_type
