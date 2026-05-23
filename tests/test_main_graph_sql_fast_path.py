import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

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


_ensure_package("agents", ROOT / "agents")
_ensure_package("agents.orchestrator", ROOT / "agents" / "orchestrator")
_ensure_package("agents.orchestrator.graph", ROOT / "agents" / "orchestrator" / "graph")

_load_module("agents.orchestrator.llm", ROOT / "agents" / "orchestrator" / "llm.py")
_load_module("agents.orchestrator.graph.state", ROOT / "agents" / "orchestrator" / "graph" / "state.py")
_load_module("agents.orchestrator.graph.planner", ROOT / "agents" / "orchestrator" / "graph" / "planner.py")
_load_module("agents.orchestrator.graph.sql_logging", ROOT / "agents" / "orchestrator" / "graph" / "sql_logging.py")
_load_module("agents.orchestrator.graph.sql_agent", ROOT / "agents" / "orchestrator" / "graph" / "sql_agent.py")
MAIN_GRAPH_MODULE = _load_module(
    "agents.orchestrator.graph.main_graph",
    ROOT / "agents" / "orchestrator" / "graph" / "main_graph.py",
)


HumanMessage = MAIN_GRAPH_MODULE.HumanMessage


class MainGraphSqlFastPathTests(unittest.TestCase):
    def setUp(self):
        self.original_execute_query = MAIN_GRAPH_MODULE.execute_query

    def tearDown(self):
        MAIN_GRAPH_MODULE.execute_query = self.original_execute_query

    def test_direct_sql_fast_path_handles_latest_expense_query(self):
        async def fake_execute_query(sql: str):
            self.assertIn("ORDER BY transaction_date DESC LIMIT 1", sql)
            return [
                {
                    "amount": 500000,
                    "category_name": "Thể thao",
                    "description": "vợt cầu lông",
                    "transaction_date": "2026-05-21T15:20:00+00:00",
                }
            ]

        MAIN_GRAPH_MODULE.execute_query = fake_execute_query

        result = asyncio.run(
            MAIN_GRAPH_MODULE._try_direct_sql_report(
                {
                    "messages": [HumanMessage(content="lần chi tiêu gần nhất")],
                    "user_id": "af1eb20b-54e5-4910-b974-60151185e48f",
                }
            )
        )

        self.assertIsInstance(result, AIMessage)
        self.assertIn("| Số tiền | Danh mục | Mô tả | Thời gian |", result.content)
        self.assertIn("| 500,000 ₫ | Thể thao | vợt cầu lông |", result.content)

    def test_direct_sql_fast_path_handles_last_week_total_query(self):
        async def fake_execute_query(sql: str):
            self.assertIn("SUM(amount)", sql)
            self.assertIn("INTERVAL '7 days'", sql)
            return [{"total_amount": 1250000}]

        MAIN_GRAPH_MODULE.execute_query = fake_execute_query

        result = asyncio.run(
            MAIN_GRAPH_MODULE._try_direct_sql_report(
                {
                    "messages": [HumanMessage(content="tổng chi tuần vừa rồi")],
                    "user_id": "af1eb20b-54e5-4910-b974-60151185e48f",
                }
            )
        )

        self.assertIsInstance(result, AIMessage)
        self.assertIn("| Khoảng thời gian | Tổng chi |", result.content)
        self.assertIn("| 7 ngày qua | 1,250,000 ₫ |", result.content)

    def test_direct_sql_fast_path_handles_numbered_latest_expense_query(self):
        async def fake_execute_query(sql: str):
            self.assertIn("ORDER BY transaction_date DESC LIMIT 5", sql)
            return [
                {
                    "amount": 34000,
                    "category_name": "Food",
                    "description": "bun mam",
                    "transaction_date": "2026-05-21T15:37:52+00:00",
                },
                {
                    "amount": 500000,
                    "category_name": "Thể thao",
                    "description": "vợt cầu lông",
                    "transaction_date": "2026-05-20T10:00:00+00:00",
                },
            ]

        MAIN_GRAPH_MODULE.execute_query = fake_execute_query

        result = asyncio.run(
            MAIN_GRAPH_MODULE._try_direct_sql_report(
                {
                    "messages": [HumanMessage(content="5 lần chi tiêu gần nhất")],
                    "user_id": "af1eb20b-54e5-4910-b974-60151185e48f",
                }
            )
        )

        self.assertIsInstance(result, AIMessage)
        self.assertIn("| # | Số tiền | Danh mục | Mô tả | Thời gian |", result.content)
        self.assertIn("| 1 | 34,000 ₫ | Food | bun mam |", result.content)
        self.assertIn("| 2 | 500,000 ₫ | Thể thao | vợt cầu lông |", result.content)

    def test_direct_sql_fast_path_handles_today_expense_breakdown_query(self):
        async def fake_execute_query(sql: str):
            self.assertIn("DATE(transaction_date AT TIME ZONE 'Asia/Ho_Chi_Minh')", sql)
            self.assertIn("DATE(NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')", sql)
            self.assertIn("ORDER BY transaction_date DESC LIMIT 10", sql)
            return [
                {
                    "amount": 150,
                    "category_name": "Ăn uống",
                    "description": "Bữa trưa",
                    "transaction_date": "2026-05-21 12:15:00+07",
                },
                {
                    "amount": 45,
                    "category_name": "Giao thông",
                    "description": "Xăng xe",
                    "transaction_date": "2026-05-21 08:30:00+07",
                },
                {
                    "amount": 20,
                    "category_name": "Giải trí",
                    "description": "Phim ảnh",
                    "transaction_date": "2026-05-21 20:00:00+07",
                },
            ]

        MAIN_GRAPH_MODULE.execute_query = fake_execute_query

        result = asyncio.run(
            MAIN_GRAPH_MODULE._try_direct_sql_report(
                {
                    "messages": [HumanMessage(content="chi tiêu ngày hôm nay cho những việc gì")],
                    "user_id": "af1eb20b-54e5-4910-b974-60151185e48f",
                }
            )
        )

        self.assertIsInstance(result, AIMessage)
        self.assertIn("| Số tiền | Danh mục | Mô tả | Thời gian |", result.content)
        self.assertIn("| 150 ₫ | Ăn uống | Bữa trưa | 2026-05-21 12:15:00+07 |", result.content)
        self.assertIn("| 45 ₫ | Giao thông | Xăng xe | 2026-05-21 08:30:00+07 |", result.content)
        self.assertIn("| 20 ₫ | Giải trí | Phim ảnh | 2026-05-21 20:00:00+07 |", result.content)
        self.assertIn("| Tổng chi hôm nay | 215 ₫ |", result.content)


if __name__ == "__main__":
    unittest.main()
