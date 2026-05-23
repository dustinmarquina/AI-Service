import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


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


_ensure_package("agents", ROOT / "agents")
_ensure_package("agents.orchestrator", ROOT / "agents" / "orchestrator")
_ensure_package("agents.orchestrator.graph", ROOT / "agents" / "orchestrator" / "graph")

TOOL_REGISTRY_MODULE = _load_module(
    "agents.orchestrator.graph.tool_registry",
    ROOT / "agents" / "orchestrator" / "graph" / "tool_registry.py",
)


call_mcp_tool = TOOL_REGISTRY_MODULE.call_mcp_tool


class _FakeTool:
    def __init__(self, name: str, result):
        self.name = name
        self.result = result

    async def ainvoke(self, _args):
        return self.result


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.original_tools = dict(TOOL_REGISTRY_MODULE._tools)
        TOOL_REGISTRY_MODULE._tools.clear()

    def tearDown(self):
        TOOL_REGISTRY_MODULE._tools.clear()
        TOOL_REGISTRY_MODULE._tools.update(self.original_tools)

    def test_call_mcp_tool_unwraps_json_text_content_list(self):
        TOOL_REGISTRY_MODULE._tools["get_user_id"] = _FakeTool(
            "get_user_id",
            [
                {
                    "type": "text",
                    "text": '{"status":"success","user_id":"abc-123"}',
                    "id": "content-1",
                }
            ],
        )

        result = asyncio.run(call_mcp_tool("get_user_id", {}))

        self.assertEqual(
            result,
            {"status": "success", "user_id": "abc-123"},
        )


if __name__ == "__main__":
    unittest.main()
