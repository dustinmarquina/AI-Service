import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from fastapi.security import HTTPAuthorizationCredentials
from langchain_core.messages import AIMessage
from langgraph.types import Command


ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("models", ROOT / "models")
_ensure_package("routes", ROOT / "routes")

_load_module("models.schemas", ROOT / "models" / "schemas.py")
CHAT_MODULE = _load_module("routes.chat", ROOT / "routes" / "chat.py")

ChatRequest = CHAT_MODULE.ChatRequest
chat_message = CHAT_MODULE.chat_message


class _FakeGraph:
    def __init__(self, updates, interrupts=()):
        self.updates = updates
        self.interrupts = interrupts
        self.stream_mode = None
        self.config = None
        self.inputs = []

    async def aget_state(self, config, *, subgraphs=False):
        self.config = config
        return types.SimpleNamespace(interrupts=self.interrupts)

    async def astream(self, _payload, config=None, stream_mode=None):
        self.config = config
        self.stream_mode = stream_mode
        self.inputs.append(_payload)
        for update in self.updates:
            yield update


class ChatRouteStreamingTests(unittest.TestCase):
    TEST_JWT = (
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiJhZjFlYjIwYi01NGU1LTQ5MTAtYjk3NC02MDE1MTE4NWU0OGYiLCJyb2xlcyI6IlJPTEVfVVNFUiJ9."
        "signature"
    )

    def test_chat_route_returns_sse_stream(self):
        fake_graph = _FakeGraph(
            [
                {"chat": {"messages": [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])]}},
                {"chat": {"messages": [AIMessage(content="Xin chao")]}},
                {"finalize": {"response": AIMessage(content="Xin chao")}},
            ]
        )
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(main_graph=fake_graph)
            )
        )
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=self.TEST_JWT,
        )

        response = asyncio.run(
            chat_message(
                ChatRequest(message="hi"),
                request,
                credentials,
            )
        )

        self.assertEqual(response.media_type, "text/event-stream")

        async def _read_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(_read_body())
        self.assertEqual(fake_graph.stream_mode, "updates")
        self.assertEqual(
            fake_graph.config,
            {"configurable": {"thread_id": "<SESSION_ID>"}},
        )
        self.assertEqual(
            fake_graph.inputs[0]["user_id"],
            "af1eb20b-54e5-4910-b974-60151185e48f",
        )
        self.assertEqual(
            body,
            f"data: {json.dumps({'text': 'Xin chao'}, ensure_ascii=False)}\n\n",
        )

    def test_chat_route_streams_interrupt_prompt(self):
        fake_graph = _FakeGraph(
            [
                {
                    "__interrupt__": (
                        types.SimpleNamespace(value="Bạn có các ví sau, chọn ví muốn dùng:"),
                    )
                }
            ]
        )
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(main_graph=fake_graph)
            )
        )
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="test-token",
        )

        response = asyncio.run(
            chat_message(
                ChatRequest(message="500k vợt cầu lông"),
                request,
                credentials,
            )
        )

        async def _read_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(_read_body())

        self.assertEqual(
            body,
            f"data: {json.dumps({'text': 'Bạn có các ví sau, chọn ví muốn dùng:'}, ensure_ascii=False)}\n\n",
        )

    def test_chat_route_resumes_pending_interrupt_with_user_reply(self):
        fake_graph = _FakeGraph(
            [{"finalize": {"response": AIMessage(content="Đã ghi nhận chi tiêu.")}}],
            interrupts=(types.SimpleNamespace(value="pending", id="interrupt-1"),),
        )
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(main_graph=fake_graph)
            )
        )
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="test-token",
        )

        response = asyncio.run(
            chat_message(
                ChatRequest(message="1"),
                request,
                credentials,
            )
        )

        async def _read_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(_read_body())

        self.assertIsInstance(fake_graph.inputs[0], Command)
        self.assertEqual(fake_graph.inputs[0].resume, "1")
        self.assertEqual(
            body,
            f"data: {json.dumps({'text': 'Đã ghi nhận chi tiêu.'}, ensure_ascii=False)}\n\n",
        )


if __name__ == "__main__":
    unittest.main()
