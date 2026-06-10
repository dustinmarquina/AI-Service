import asyncio
import importlib.util
import os
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


_ensure_package("agents", ROOT / "agents")
_ensure_package("agents.orchestrator", ROOT / "agents" / "orchestrator")
_ensure_package("agents.orchestrator.graph", ROOT / "agents" / "orchestrator" / "graph")

DB_MODULE = _load_module(
    "agents.orchestrator.graph.db",
    ROOT / "agents" / "orchestrator" / "graph" / "db.py",
)


class _FakeCollection:
    def __init__(self, docs):
        self.docs = list(docs)

    def find(self, query):
        results = []
        for doc in self.docs:
            if all(doc.get(key) == value for key, value in query.items()):
                results.append(dict(doc))
        return results


class GraphDbTests(unittest.TestCase):
    def test_execute_mongo_query_filters_wallets_by_name_and_user(self):
        wallets = _FakeCollection(
            [
                {"walletId": "wallet-1", "userId": "user-123", "name": "du học", "active": True, "balance": 1000, "currency": "VND"},
                {"walletId": "wallet-2", "userId": "user-123", "name": "MB Bank", "active": True, "balance": 2000, "currency": "VND"},
                {"walletId": "wallet-3", "userId": "other-user", "name": "MB Bank", "active": True, "balance": 3000, "currency": "VND"},
            ]
        )

        with patch.object(DB_MODULE, "_get_mongo_collection", return_value=wallets):
            rows = asyncio.run(
                DB_MODULE._execute_mongo_query(
                    "SELECT id FROM wallets WHERE name = 'MB Bank' AND user_id = 'user-123' AND active = true LIMIT 1"
                )
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "wallet-2")

    def test_execute_mongo_query_honors_order_by_before_limit(self):
        wallets = _FakeCollection(
            [
                {"walletId": "wallet-1", "userId": "user-123", "name": "du học", "active": True, "balance": 400000000, "currency": "VND"},
                {"walletId": "wallet-2", "userId": "user-123", "name": "MB Bank", "active": True, "balance": 50000000, "currency": "VND"},
                {"walletId": "wallet-3", "userId": "user-123", "name": "Cash", "active": True, "balance": 250000, "currency": "VND"},
            ]
        )

        with patch.object(DB_MODULE, "_get_mongo_collection", return_value=wallets):
            rows = asyncio.run(
                DB_MODULE._execute_mongo_query(
                    "SELECT id FROM wallets WHERE user_id = 'user-123' ORDER BY balance ASC LIMIT 1"
                )
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "wallet-3")


if __name__ == "__main__":
    unittest.main()
