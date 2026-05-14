import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from fastapi.security import HTTPAuthorizationCredentials
from langchain_core.messages import AIMessage


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
    def __init__(self, updates):
        self.updates = updates
        self.stream_mode = None

    async def astream(self, _payload, stream_mode=None):
        self.stream_mode = stream_mode
        for update in self.updates:
            yield update


class ChatRouteStreamingTests(unittest.TestCase):
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
            credentials="test-token",
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
            body,
            f"data: {json.dumps({'text': 'Xin chao'}, ensure_ascii=False)}\n\n",
        )


if __name__ == "__main__":
    unittest.main()
