import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

MODULE_PATH = Path(__file__).resolve().parents[1] / "agents" / "orchestrator" / "graph" / "sql_logging.py"
MODULE_SPEC = spec_from_file_location("sql_logging", MODULE_PATH)
SQL_LOGGING_MODULE = module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(SQL_LOGGING_MODULE)
extract_faulty_sql_from_messages = SQL_LOGGING_MODULE.extract_faulty_sql_from_messages


class ExtractFaultySqlTests(unittest.TestCase):
    def test_returns_failed_sql_query(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_query",
                        "id": "call-1",
                        "args": {"query": "SELECT total FROM transactions WHERE bad_column = 1"},
                    }
                ],
            ),
            ToolMessage(
                content='Error: column "bad_column" does not exist',
                tool_call_id="call-1",
                status="error",
            ),
        ]

        self.assertEqual(
            extract_faulty_sql_from_messages(messages),
            "SELECT total FROM transactions WHERE bad_column = 1",
        )

    def test_ignores_successful_tool_results(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_query",
                        "id": "call-1",
                        "args": {"query": "SELECT total FROM transactions LIMIT 5"},
                    }
                ],
            ),
            ToolMessage(
                content="[('ok',)]",
                tool_call_id="call-1",
                status="success",
            ),
        ]

        self.assertIsNone(extract_faulty_sql_from_messages(messages))


if __name__ == "__main__":
    unittest.main()
