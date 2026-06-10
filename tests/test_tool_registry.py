import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL_REGISTRY_MODULE = _load_module(
    "agents.orchestrator.graph.tool_registry",
    ROOT / "agents" / "orchestrator" / "graph" / "tool_registry.py",
)


class _FakeField:
    def __init__(self, annotation, required=True):
        self.annotation = annotation
        self._required = required

    def is_required(self):
        return self._required


class _FakeSchema:
    model_fields = {
        "amount": _FakeField(str),
        "description": _FakeField(str),
        "wallet_id": _FakeField(str, required=False),
    }


class _FakeTool:
    def __init__(self):
        self.name = "create_transaction"
        self.args_schema = _FakeSchema
        self.parameters = None
        self.seen_args = None

    async def ainvoke(self, args):
        self.seen_args = dict(args)
        return {"status": "success", "data": {"id": "txn-1", "amount": args["amount"]}}


class _JsonSchemaTool:
    def __init__(self):
        self.name = "create_transaction"
        self.args_schema = None
        self.parameters = {
            "type": "object",
            "properties": {
                "amount": {"type": "string"},
                "description": {"type": "string"},
                "wallet_id": {"type": "string"},
            },
            "required": ["amount", "description"],
        }
        self.seen_args = None

    async def ainvoke(self, args):
        self.seen_args = dict(args)
        return {"status": "success", "data": {"id": "txn-2", "amount": args["amount"]}}


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        TOOL_REGISTRY_MODULE._tools.clear()

    def tearDown(self):
        TOOL_REGISTRY_MODULE._tools.clear()

    def test_call_mcp_tool_coerces_string_fields_from_numeric_inputs(self):
        tool = _FakeTool()
        TOOL_REGISTRY_MODULE.register_tools([tool])

        result = asyncio.run(
            TOOL_REGISTRY_MODULE.call_mcp_tool(
                "create_transaction",
                {"amount": 50000, "description": "bun bo hue"},
            )
        )

        self.assertEqual(tool.seen_args["amount"], "50000")
        self.assertEqual(result["data"]["amount"], "50000")

    def test_call_mcp_tool_coerces_string_fields_from_json_schema_parameters(self):
        tool = _JsonSchemaTool()
        TOOL_REGISTRY_MODULE.register_tools([tool])

        result = asyncio.run(
            TOOL_REGISTRY_MODULE.call_mcp_tool(
                "create_transaction",
                {"amount": 50000, "description": "my quang"},
            )
        )

        self.assertEqual(tool.seen_args["amount"], "50000")
        self.assertEqual(result["data"]["amount"], "50000")


if __name__ == "__main__":
    unittest.main()
