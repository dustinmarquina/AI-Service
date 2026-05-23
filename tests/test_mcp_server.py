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

_load_module(
    "agents.orchestrator.graph.state",
    ROOT / "agents" / "orchestrator" / "graph" / "state.py",
)
MCP_SERVER_MODULE = _load_module(
    "agents.orchestrator.mcp_server",
    ROOT / "agents" / "orchestrator" / "mcp_server.py",
)


class McpServerTests(unittest.TestCase):
    def setUp(self):
        self.original_api_call = MCP_SERVER_MODULE._api_call

    def tearDown(self):
        MCP_SERVER_MODULE._api_call = self.original_api_call

    def test_create_transaction_accepts_wallet_id_and_category_name(self):
        captured = {}

        async def fake_api_call(method, url, *, bearer="", payload=None, params=None):
            captured["method"] = method
            captured["url"] = url
            captured["bearer"] = bearer
            captured["payload"] = payload
            captured["params"] = params
            return {"status": "success", "data": {"id": "txn-1"}}

        MCP_SERVER_MODULE._api_call = fake_api_call

        result = asyncio.run(
            MCP_SERVER_MODULE.create_transaction(
                amount="500k",
                description="vợt cầu lông",
                wallet_id="wallet-123",
                category_name="Thể thao",
                token="abc-token",
                user_id="user-123",
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], MCP_SERVER_MODULE.TRANSACTION_API_URL)
        self.assertEqual(captured["bearer"], "abc-token")
        self.assertEqual(
            captured["payload"]["walletId"],
            "wallet-123",
        )
        self.assertEqual(
            captured["payload"]["categoryName"],
            "Thể thao",
        )


if __name__ == "__main__":
    unittest.main()
