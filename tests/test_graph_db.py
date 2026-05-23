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


class GraphDbTests(unittest.TestCase):
    def test_build_database_dsn_adds_database_to_server_base_uri(self):
        with patch.dict(
            os.environ,
            {"POSTGRES_URI": "postgresql://user:pass@db.example.com"},
            clear=False,
        ):
            dsn = DB_MODULE._build_database_dsn("wallet_db")

        self.assertEqual(
            dsn,
            "postgresql://user:pass@db.example.com/wallet_db",
        )

    def test_build_database_dsn_replaces_existing_database_path(self):
        with patch.dict(
            os.environ,
            {"POSTGRES_URI": "postgresql://user:pass@db.example.com/postgres"},
            clear=False,
        ):
            dsn = DB_MODULE._build_database_dsn("transaction_db")

        self.assertEqual(
            dsn,
            "postgresql://user:pass@db.example.com/transaction_db",
        )

    def test_choose_database_for_wallets_query(self):
        database_name = DB_MODULE._choose_database_name_for_sql(
            "SELECT id, name FROM wallets WHERE active = true"
        )

        self.assertEqual(database_name, "wallet_db")

    def test_choose_database_for_categories_query(self):
        database_name = DB_MODULE._choose_database_name_for_sql(
            "SELECT id FROM categories WHERE user_id = 'abc'"
        )

        self.assertEqual(database_name, "transaction_db")

    def test_choose_database_for_cross_database_query_raises(self):
        with self.assertRaises(ValueError):
            DB_MODULE._choose_database_name_for_sql(
                "SELECT * FROM wallets JOIN transactions ON wallets.id = transactions.wallet_id"
            )


if __name__ == "__main__":
    unittest.main()
