import asyncio
import json
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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


class _FakeMCP:
    def __init__(self, name=""):
        self.name = name

    def tool(self):
        def decorator(func):
            return func
        return decorator


_ensure_package("agents", ROOT / "agents")
_ensure_package("agents.orchestrator", ROOT / "agents" / "orchestrator")
_ensure_package("agents.orchestrator.graph", ROOT / "agents" / "orchestrator" / "graph")

sys.modules["agents.orchestrator.graph.state"] = types.SimpleNamespace(State=dict)
sys.modules["fastmcp"] = types.SimpleNamespace(FastMCP=_FakeMCP)

MCP_SERVER_MODULE = _load_module(
    "agents.orchestrator.mcp_server",
    ROOT / "agents" / "orchestrator" / "mcp_server.py",
)


class _InsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class _FakeCollection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])

    def find_one(self, query):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return dict(doc)
        return None

    def find(self, query):
        return [dict(doc) for doc in self.docs if all(doc.get(k) == v for k, v in query.items())]

    def insert_one(self, document):
        inserted = dict(document)
        inserted.setdefault("_id", f"id-{len(self.docs) + 1}")
        self.docs.append(inserted)
        return _InsertResult(inserted["_id"])


class _StaticLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return types.SimpleNamespace(content=self.content)


class McpServerMongoTests(unittest.TestCase):
    def test_create_wallet_inserts_manual_wallet_with_defaults(self):
        wallets = _FakeCollection()

        def fake_get_collection(name):
            return {"wallets": wallets}[name]

        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_wallet(
                    name="Cash",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(wallets.docs[0]["name"], "Cash")
        self.assertEqual(wallets.docs[0]["balance"], 0)
        self.assertEqual(wallets.docs[0]["currency"], "VND")
        self.assertEqual(wallets.docs[0]["walletType"], "MANUAL")
        self.assertEqual(wallets.docs[0]["userId"], "user-123")
        self.assertIsInstance(result["data"]["createdAt"], str)
        json.dumps(result)

    def test_create_transaction_inserts_into_transactions_collection(self):
        transactions = _FakeCollection()
        wallets = _FakeCollection([{"_id": "wallet-1", "userId": "user-123", "name": "Cash"}])

        def fake_get_collection(name):
            return {"transactions": transactions, "wallets": wallets}[name]

        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_transaction(
                    amount="50k",
                    description="cafe",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(transactions.docs[0]["amount"], 50000)
        self.assertEqual(transactions.docs[0]["wallet_id"], "wallet-1")
        self.assertEqual(transactions.docs[0]["user_id"], "user-123")
        self.assertEqual(transactions.docs[0]["type"], "EXPENSE")
        self.assertIn("id", transactions.docs[0])
        self.assertIn("transaction_date", transactions.docs[0])
        self.assertIn("created_at", transactions.docs[0])
        self.assertIn("updated_at", transactions.docs[0])
        self.assertIsNone(transactions.docs[0]["image_url"])
        self.assertNotIn("walletId", transactions.docs[0])
        self.assertNotIn("userId", transactions.docs[0])
        self.assertIsInstance(result["data"]["created_at"], str)
        json.dumps(result)

    def test_create_transaction_requests_wallet_selection_for_multiple_wallets(self):
        transactions = _FakeCollection()
        wallets = _FakeCollection(
            [
                {"_id": "wallet-1", "walletId": "wallet-1", "userId": "user-123", "name": "Cash", "balance": 1000, "currency": "VND"},
                {"_id": "wallet-2", "walletId": "wallet-2", "userId": "user-123", "name": "Bank", "balance": 2000, "currency": "VND"},
            ]
        )

        def fake_get_collection(name):
            return {"transactions": transactions, "wallets": wallets}[name]

        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_transaction(
                    amount="50k",
                    description="bun cha",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["selection_field"], "wallet_id")
        self.assertEqual(result["value_field"], "walletId")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(transactions.docs, [])

    def test_create_transaction_uses_explicit_wallet_name_when_it_matches_exactly(self):
        transactions = _FakeCollection()
        wallets = _FakeCollection(
            [
                {"_id": "wallet-1", "walletId": "wallet-1", "userId": "user-123", "name": "du học", "balance": 1000, "currency": "VND"},
                {"_id": "wallet-2", "walletId": "wallet-2", "userId": "user-123", "name": "MB Bank", "balance": 2000, "currency": "VND"},
            ]
        )

        def fake_get_collection(name):
            return {"transactions": transactions, "wallets": wallets}[name]

        llm = _StaticLLM(
            '{"route":"execute","reason":"MB Bank is the named wallet.","bindings":{"wallet_id":"wallet-2"}}'
        )
        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection), \
             patch.object(MCP_SERVER_MODULE, "_get_candidate_binding_llm", return_value=llm):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_transaction(
                    amount="50k",
                    description="bun cha",
                    wallet_name="MB Bank",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(transactions.docs[0]["wallet_id"], "wallet-2")
        self.assertEqual(transactions.docs[0]["amount"], 50000)
        self.assertEqual(len(llm.calls), 1)

    def test_create_transaction_uses_request_text_to_match_wallet_name(self):
        transactions = _FakeCollection()
        wallets = _FakeCollection(
            [
                {"_id": "wallet-1", "walletId": "wallet-1", "userId": "user-123", "name": "du học", "balance": 1000, "currency": "VND"},
                {"_id": "wallet-2", "walletId": "wallet-2", "userId": "user-123", "name": "MB Bank", "balance": 2000, "currency": "VND"},
            ]
        )

        def fake_get_collection(name):
            return {"transactions": transactions, "wallets": wallets}[name]

        llm = _StaticLLM(
            '{"route":"execute","reason":"MB Bank is referenced in the request text.","bindings":{"wallet_id":"wallet-2"}}'
        )
        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection), \
             patch.object(MCP_SERVER_MODULE, "_get_candidate_binding_llm", return_value=llm):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_transaction(
                    amount="50k",
                    description="bun cha",
                    request_text="50k bun cha vao vi MB Bank",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(transactions.docs[0]["wallet_id"], "wallet-2")
        self.assertEqual(len(llm.calls), 1)

    def test_create_transaction_uses_partial_wallet_name_when_unique(self):
        transactions = _FakeCollection()
        wallets = _FakeCollection(
            [
                {"_id": "wallet-1", "walletId": "wallet-1", "userId": "user-123", "name": "du học", "balance": 1000, "currency": "VND"},
                {"_id": "wallet-2", "walletId": "wallet-2", "userId": "user-123", "name": "MB Bank", "balance": 2000, "currency": "VND"},
            ]
        )

        def fake_get_collection(name):
            return {"transactions": transactions, "wallets": wallets}[name]

        llm = _StaticLLM(
            '{"route":"execute","reason":"MB uniquely identifies MB Bank among the candidates.","bindings":{"wallet_id":"wallet-2"}}'
        )
        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection), \
             patch.object(MCP_SERVER_MODULE, "_get_candidate_binding_llm", return_value=llm):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_transaction(
                    amount="50k",
                    description="bun cha",
                    request_text="50k bun cha vao vi MB",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(transactions.docs[0]["wallet_id"], "wallet-2")
        self.assertEqual(len(llm.calls), 1)

    def test_create_transaction_requests_wallet_selection_when_llm_cannot_resolve_candidate(self):
        transactions = _FakeCollection()
        wallets = _FakeCollection(
            [
                {"_id": "wallet-1", "walletId": "wallet-1", "userId": "user-123", "name": "du học", "balance": 1000, "currency": "VND"},
                {"_id": "wallet-2", "walletId": "wallet-2", "userId": "user-123", "name": "MB Bank", "balance": 2000, "currency": "VND"},
            ]
        )

        def fake_get_collection(name):
            return {"transactions": transactions, "wallets": wallets}[name]

        llm = _StaticLLM(
            '{"route":"execute","reason":"invalid candidate","bindings":{"wallet_id":"wallet-999"}}'
        )
        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection), \
             patch.object(MCP_SERVER_MODULE, "_get_candidate_binding_llm", return_value=llm):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_transaction(
                    amount="50k",
                    description="bun cha",
                    request_text="50k bun cha vao vi MB Bank",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(transactions.docs, [])

    def test_create_transaction_uses_lowest_balance_wallet_deterministically(self):
        transactions = _FakeCollection()
        wallets = _FakeCollection(
            [
                {"_id": "wallet-1", "walletId": "wallet-1", "userId": "user-123", "name": "Du học", "balance": 400000000, "currency": "VND"},
                {"_id": "wallet-2", "walletId": "wallet-2", "userId": "user-123", "name": "MB Bank", "balance": 50000000, "currency": "VND"},
                {"_id": "wallet-3", "walletId": "wallet-3", "userId": "user-123", "name": "Cash", "balance": 250000, "currency": "VND"},
            ]
        )

        def fake_get_collection(name):
            return {"transactions": transactions, "wallets": wallets}[name]

        llm = _StaticLLM(
            '{"route":"execute","reason":"wrong on purpose","bindings":{"wallet_id":"wallet-1"}}'
        )
        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection), \
             patch.object(MCP_SERVER_MODULE, "_get_candidate_binding_llm", return_value=llm):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_transaction(
                    amount="50k",
                    description="bun bo hue",
                    request_text="50k bun bo hue vao vi co so du thap nhat",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(transactions.docs[0]["wallet_id"], "wallet-3")
        self.assertEqual(len(llm.calls), 0)

    def test_create_category_inserts_into_categories_collection(self):
        categories = _FakeCollection()

        def fake_get_collection(name):
            return {"categories": categories}[name]

        with patch.object(MCP_SERVER_MODULE, "_get_collection", side_effect=fake_get_collection):
            result = asyncio.run(
                MCP_SERVER_MODULE.create_category(
                    name="Coffee",
                    budget_limit="100k",
                    user_id="user-123",
                )
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(categories.docs[0]["name"], "Coffee")
        self.assertEqual(categories.docs[0]["budgetLimit"], 100000)
        self.assertEqual(categories.docs[0]["userId"], "user-123")
        self.assertIsInstance(result["data"]["createdAt"], str)
        json.dumps(result)

    def test_get_wallet_summary_reads_wallets_from_mongo(self):
        wallets = _FakeCollection(
            [
                {"_id": "wallet-1", "userId": "user-123", "name": "Cash", "balance": 1000, "currency": "VND"},
                {"_id": "wallet-2", "userId": "user-123", "name": "Bank", "balance": 2500, "currency": "VND"},
            ]
        )

        with patch.object(MCP_SERVER_MODULE, "_get_collection", return_value=wallets):
            result = asyncio.run(MCP_SERVER_MODULE.get_wallet_summary(user_id="user-123"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["wallet_summary"]["totalBalance"], 3500)
        self.assertEqual(len(result["wallet_summary"]["wallets"]), 2)


if __name__ == "__main__":
    unittest.main()
