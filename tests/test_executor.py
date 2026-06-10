import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


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


class _FakeAIMessage:
    def __init__(self, content="", **_kwargs):
        self.content = content


class _FakeSystemMessage:
    def __init__(self, content="", **_kwargs):
        self.content = content


sys.modules["langchain_core.messages"] = types.SimpleNamespace(
    BaseMessage=object,
    AIMessage=_FakeAIMessage,
    SystemMessage=_FakeSystemMessage,
)
sys.modules["langgraph.graph.message"] = types.SimpleNamespace(add_messages=lambda x, y: x + y)
sys.modules["langgraph.types"] = types.SimpleNamespace(interrupt=lambda value: value)

_load_module("agents.orchestrator.graph.state", ROOT / "agents" / "orchestrator" / "graph" / "state.py")
EXECUTOR_MODULE = _load_module(
    "agents.orchestrator.graph.executor",
    ROOT / "agents" / "orchestrator" / "graph" / "executor.py",
)


class ExecutorTests(unittest.TestCase):
    def test_sql_step_records_past_step_on_success(self):
        fake_db = types.SimpleNamespace(execute_query=AsyncMock(return_value=[{"id": "wallet-1", "name": "Cash"}]))
        sys.modules["agents.orchestrator.graph.db"] = fake_db

        state = {
            "steps": [
                {
                    "id": "s1",
                    "type": "sql",
                    "reasoning": "Need wallet rows before choosing one",
                    "query_hint": "SELECT id, name FROM wallets WHERE user_id = :user_id",
                },
            ],
            "step_index": 0,
            "step_results": {},
            "past_steps": [],
            "user_id": "user-123",
        }

        result = asyncio.run(EXECUTOR_MODULE.executor_node(state))

        self.assertEqual(result["step_index"], 1)
        self.assertEqual(result["step_results"]["s1"]["id"], "wallet-1")
        self.assertEqual(len(result["past_steps"]), 1)
        self.assertEqual(result["past_steps"][0]["step_id"], "s1")
        self.assertEqual(result["past_steps"][0]["type"], "sql")
        self.assertEqual(result["past_steps"][0]["status"], "success")
        self.assertEqual(result["past_steps"][0]["reasoning"], "Need wallet rows before choosing one")

    def test_sql_step_with_multiple_rows_does_not_interrupt_without_selection_mode(self):
        fake_db = types.SimpleNamespace(
            execute_query=AsyncMock(
                return_value=[
                    {"id": "txn-1", "description": "bun cha", "amount": 50000},
                    {"id": "txn-2", "description": "pho", "amount": 60000},
                ]
            )
        )
        sys.modules["agents.orchestrator.graph.db"] = fake_db

        original_interrupt = EXECUTOR_MODULE.interrupt
        EXECUTOR_MODULE.interrupt = lambda value: (_ for _ in ()).throw(AssertionError(f"unexpected interrupt: {value}"))
        try:
            state = {
                "steps": [
                    {
                        "id": "s1",
                        "type": "sql",
                        "reasoning": "Need recent transactions to answer the question",
                        "query_hint": "SELECT id, description, amount FROM transactions WHERE user_id = :user_id",
                    },
                ],
                "step_index": 0,
                "step_results": {},
                "past_steps": [],
                "user_id": "user-123",
            }

            result = asyncio.run(EXECUTOR_MODULE.executor_node(state))
        finally:
            EXECUTOR_MODULE.interrupt = original_interrupt

        self.assertEqual(result["step_index"], 1)
        self.assertEqual(len(result["step_results"]["s1"]), 2)
        self.assertEqual(result["past_steps"][0]["summary"], "SQL step s1 returned 2 rows")

    def test_sql_step_resolves_embedded_placeholders_in_query_text(self):
        fake_db = types.SimpleNamespace(execute_query=AsyncMock(return_value=[{"id": "wallet-2"}]))
        sys.modules["agents.orchestrator.graph.db"] = fake_db

        state = {
            "steps": [
                {
                    "id": "s1",
                    "type": "sql",
                    "reasoning": "Need the MB Bank wallet id for the transaction",
                    "query_hint": "SELECT id FROM wallets WHERE name = 'MB Bank' AND user_id = $s0.user_id AND active = true LIMIT 1",
                    "selection_mode": "required",
                },
            ],
            "step_index": 0,
            "step_results": {
                "s0": {"status": "success", "user_id": "user-123"},
            },
            "past_steps": [],
            "user_id": "user-123",
        }

        result = asyncio.run(EXECUTOR_MODULE.executor_node(state))

        fake_db.execute_query.assert_awaited_once_with(
            "SELECT id FROM wallets WHERE name = 'MB Bank' AND user_id = 'user-123' AND active = true LIMIT 1"
        )
        self.assertEqual(result["step_results"]["s1"]["id"], "wallet-2")

    def test_tool_step_records_past_step_and_injects_runtime_args(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(return_value={"status": "success", "data": {"id": "txn-1"}})
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        state = {
            "steps": [
                {
                    "id": "s1",
                    "type": "tool",
                    "name": "create_transaction",
                    "reasoning": "Need to create the transaction after parsing the user request",
                    "args": {"amount": "50k"},
                },
            ],
            "step_index": 0,
            "step_results": {},
            "past_steps": [],
            "user_id": "user-123",
            "token": "token-123",
        }

        result = asyncio.run(EXECUTOR_MODULE.executor_node(state))

        self.assertEqual(result["step_index"], 1)
        fake_registry.call_mcp_tool.assert_awaited_once_with(
            "create_transaction",
            {"amount": "50k", "token": "token-123", "user_id": "user-123"},
        )
        self.assertEqual(len(result["past_steps"]), 1)
        self.assertEqual(result["past_steps"][0]["status"], "success")
        self.assertEqual(result["past_steps"][0]["output"]["data"]["id"], "txn-1")
        self.assertEqual(
            result["past_steps"][0]["reasoning"],
            "Need to create the transaction after parsing the user request",
        )

    def test_create_transaction_receives_latest_user_text_as_request_text(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(return_value={"status": "success", "data": {"id": "txn-1"}})
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        class _FakeHumanMessage:
            def __init__(self, content):
                self.content = content

        state = {
            "steps": [
                {
                    "id": "s1",
                    "type": "tool",
                    "name": "create_transaction",
                    "args": {"amount": "50k", "description": "bún chả"},
                },
            ],
            "step_index": 0,
            "step_results": {},
            "past_steps": [],
            "user_id": "user-123",
            "messages": [_FakeHumanMessage("50k bún chả vào ví MB Bank")],
        }

        asyncio.run(EXECUTOR_MODULE.executor_node(state))

        fake_registry.call_mcp_tool.assert_awaited_once_with(
            "create_transaction",
            {
                "amount": "50k",
                "description": "bún chả",
                "user_id": "user-123",
                "request_text": "50k bún chả vào ví MB Bank",
            },
        )

    def test_tool_step_prefers_user_id_from_prior_get_user_id_result(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(return_value={"status": "success", "data": {"id": "txn-1"}})
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        state = {
            "steps": [
                {
                    "id": "s1",
                    "type": "tool",
                    "name": "create_transaction",
                    "args": {"amount": "50k", "description": "mỳ quảng"},
                },
            ],
            "step_index": 0,
            "step_results": {
                "s0": {"status": "success", "user_id": "user-123"},
            },
            "past_steps": [],
            "token": "token-123",
        }

        with patch.dict("os.environ", {"TRANSACTION_USER_ID": "wrong-user"}, clear=False):
            asyncio.run(EXECUTOR_MODULE.executor_node(state))

        fake_registry.call_mcp_tool.assert_awaited_once_with(
            "create_transaction",
            {
                "amount": "50k",
                "description": "mỳ quảng",
                "token": "token-123",
                "user_id": "user-123",
            },
        )

    def test_build_execution_summary_uses_latest_attempt_per_step_id(self):
        message = EXECUTOR_MODULE.build_execution_summary(
            {
                "past_steps": [
                    {"step_id": "s0", "status": "success", "summary": "get_user_id completed"},
                    {"step_id": "s1", "status": "success", "summary": "SQL step s1 returned 3 rows"},
                    {"step_id": "s2", "status": "error", "summary": "No wallet found for this user."},
                    {"step_id": "s2", "status": "success", "summary": "create_transaction completed"},
                ]
            }
        )

        self.assertIn("✅ get_user_id completed", message.content)
        self.assertIn("✅ SQL step s1 returned 3 rows", message.content)
        self.assertIn("✅ create_transaction completed", message.content)
        self.assertNotIn("❌ No wallet found for this user.", message.content)

    def test_tool_step_records_failure_in_past_steps(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(return_value={"status": "error", "message": "boom"})
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        state = {
            "steps": [
                {"id": "s1", "type": "tool", "name": "create_transaction", "args": {"amount": "50k"}},
            ],
            "step_index": 0,
            "step_results": {},
            "past_steps": [],
        }

        result = asyncio.run(EXECUTOR_MODULE.executor_node(state))

        self.assertEqual(result["step_index"], 1)
        self.assertEqual(result["past_steps"][0]["status"], "error")
        self.assertEqual(result["past_steps"][0]["summary"], "boom")

    def test_tool_step_interrupts_generically_when_tool_requests_input(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(
                return_value={
                    "status": "needs_input",
                    "prompt": "Choose a wallet",
                    "selection_field": "wallet_id",
                    "value_field": "walletId",
                    "candidates": [
                        {"walletId": "wallet-1", "name": "Cash", "balance": 1000, "currency": "VND"},
                        {"walletId": "wallet-2", "name": "Bank", "balance": 2000, "currency": "VND"},
                    ],
                }
            )
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        original_interrupt = EXECUTOR_MODULE.interrupt
        EXECUTOR_MODULE.interrupt = lambda value: value
        try:
            state = {
                "steps": [
                    {"id": "s1", "type": "tool", "name": "create_transaction", "args": {"amount": "50k"}},
                ],
                "step_index": 0,
                "step_results": {},
                "past_steps": [],
                "user_id": "user-123",
            }

            result = asyncio.run(EXECUTOR_MODULE.executor_node(state))
        finally:
            EXECUTOR_MODULE.interrupt = original_interrupt

        self.assertEqual(result["step_index"], 0)
        self.assertIn("s1__pending_input", result["step_results"])
        self.assertEqual(result["step_results"]["s1__pending_input"]["selection_field"], "wallet_id")
        self.assertIn("1. Cash", result["step_results"]["s1__pending_input"]["prompt"])
        self.assertIn("2. Bank", result["step_results"]["s1__pending_input"]["prompt"])

    def test_tool_step_resumes_generically_after_user_selection(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(return_value={"status": "success", "data": {"id": "txn-1"}})
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        original_interrupt = EXECUTOR_MODULE.interrupt
        EXECUTOR_MODULE.interrupt = lambda _value: "2"
        try:
            state = {
                "steps": [
                    {"id": "s1", "type": "tool", "name": "create_transaction", "args": {"amount": "50k"}},
                ],
                "step_index": 0,
                "step_results": {
                    "s1__pending_input": {
                        "prompt": "Choose a wallet",
                        "selection_field": "wallet_id",
                        "value_field": "walletId",
                        "candidates": [
                            {"walletId": "wallet-1", "name": "Cash"},
                            {"walletId": "wallet-2", "name": "Bank"},
                        ],
                        "args": {"amount": "50k", "user_id": "user-123"},
                    }
                },
                "past_steps": [],
                "user_id": "user-123",
            }

            result = asyncio.run(EXECUTOR_MODULE.executor_node(state))
        finally:
            EXECUTOR_MODULE.interrupt = original_interrupt

        fake_registry.call_mcp_tool.assert_awaited_once_with(
            "create_transaction",
            {"amount": "50k", "user_id": "user-123", "wallet_id": "wallet-2"},
        )
        self.assertEqual(result["step_index"], 1)
        self.assertEqual(result["step_results"]["s1"]["data"]["id"], "txn-1")

    def test_tool_step_pending_prompt_is_not_wrapped_repeatedly(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(return_value={"status": "success", "data": {"id": "txn-1"}})
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        captured_prompts = []
        original_interrupt = EXECUTOR_MODULE.interrupt
        EXECUTOR_MODULE.interrupt = lambda value: captured_prompts.append(value) or value
        try:
            state = {
                "steps": [
                    {"id": "s1", "type": "tool", "name": "create_transaction", "args": {"amount": "50k"}},
                ],
                "step_index": 0,
                "step_results": {
                    "s1__pending_input": {
                        "prompt": "Please choose which wallet to use for this transaction:\n  1. Cash — 1000 | VND\n  2. Bank — 2000 | VND\n\nReply with the number or the option name:",
                        "prompt_intro": "Please choose which wallet to use for this transaction:",
                        "selection_field": "wallet_id",
                        "value_field": "walletId",
                        "candidates": [
                            {"walletId": "wallet-1", "name": "Cash", "balance": 1000, "currency": "VND"},
                            {"walletId": "wallet-2", "name": "Bank", "balance": 2000, "currency": "VND"},
                        ],
                        "args": {"amount": "50k", "user_id": "user-123"},
                    }
                },
                "past_steps": [],
                "user_id": "user-123",
            }

            result = asyncio.run(EXECUTOR_MODULE.executor_node(state))
        finally:
            EXECUTOR_MODULE.interrupt = original_interrupt

        self.assertEqual(result["step_index"], 0)
        self.assertEqual(len(captured_prompts), 1)
        self.assertEqual(
            captured_prompts[0],
            "Please choose which wallet to use for this transaction:\n  1. Cash — 1000 | VND\n  2. Bank — 2000 | VND\n\nReply with the number or the option name:",
        )

    def test_resolve_row_selection_accepts_resume_mapping_values(self):
        matched = EXECUTOR_MODULE._resolve_row_selection(
            {"interrupt-1": "2"},
            [
                {"walletId": "wallet-1", "name": "Cash"},
                {"walletId": "wallet-2", "name": "Bank"},
            ],
        )

        self.assertEqual(matched["walletId"], "wallet-2")

    def test_resolve_row_selection_accepts_vietnamese_ordinal(self):
        matched = EXECUTOR_MODULE._resolve_row_selection(
            "cái thứ 2",
            [
                {"walletId": "wallet-1", "name": "Cash"},
                {"walletId": "wallet-2", "name": "Bank"},
            ],
        )

        self.assertEqual(matched["walletId"], "wallet-2")

    def test_resolve_row_selection_is_accent_insensitive(self):
        matched = EXECUTOR_MODULE._resolve_row_selection(
            "du hoc",
            [
                {"walletId": "wallet-1", "name": "du học"},
                {"walletId": "wallet-2", "name": "MB Bank"},
            ],
        )

        self.assertEqual(matched["walletId"], "wallet-1")

    def test_tool_step_uses_llm_fallback_for_fuzzy_selection_reply(self):
        fake_registry = types.SimpleNamespace(
            call_mcp_tool=AsyncMock(return_value={"status": "success", "data": {"id": "txn-1"}})
        )
        sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

        class _FakeLLM:
            async def ainvoke(self, _messages):
                return types.SimpleNamespace(
                    content='{"route":"select","index":2,"reason":"ngan hang matches MB Bank"}'
                )

        original_interrupt = EXECUTOR_MODULE.interrupt
        original_get_llm = EXECUTOR_MODULE._get_selection_llm
        EXECUTOR_MODULE.interrupt = lambda _value: "cái ngân hàng"
        EXECUTOR_MODULE._selection_llm = None
        EXECUTOR_MODULE._get_selection_llm = lambda: _FakeLLM()
        try:
            state = {
                "steps": [
                    {"id": "s1", "type": "tool", "name": "create_transaction", "args": {"amount": "50k"}},
                ],
                "step_index": 0,
                "step_results": {
                    "s1__pending_input": {
                        "prompt_intro": "Please choose which wallet to use for this transaction:",
                        "selection_field": "wallet_id",
                        "value_field": "walletId",
                        "candidates": [
                            {"walletId": "wallet-1", "name": "Cash"},
                            {"walletId": "wallet-2", "name": "MB Bank"},
                        ],
                        "args": {"amount": "50k", "user_id": "user-123"},
                    }
                },
                "past_steps": [],
                "user_id": "user-123",
            }

            result = asyncio.run(EXECUTOR_MODULE.executor_node(state))
        finally:
            EXECUTOR_MODULE.interrupt = original_interrupt
            EXECUTOR_MODULE._get_selection_llm = original_get_llm
            EXECUTOR_MODULE._selection_llm = None

        fake_registry.call_mcp_tool.assert_awaited_once_with(
            "create_transaction",
            {"amount": "50k", "user_id": "user-123", "wallet_id": "wallet-2"},
        )
        self.assertEqual(result["step_index"], 1)

    def test_sql_step_interrupts_generically_for_multiple_rows(self):
        fake_db = types.SimpleNamespace(
            execute_query=AsyncMock(
                return_value=[
                    {"id": "wallet-1", "name": "Cash", "balance": 1000, "currency": "VND"},
                    {"id": "wallet-2", "name": "Bank", "balance": 2000, "currency": "VND"},
                ]
            )
        )
        sys.modules["agents.orchestrator.graph.db"] = fake_db

        original_interrupt = EXECUTOR_MODULE.interrupt
        EXECUTOR_MODULE.interrupt = lambda value: value
        try:
            state = {
                "steps": [
                    {
                        "id": "s1",
                        "type": "sql",
                        "selection_mode": "required",
                        "query_hint": "SELECT id, name FROM wallets WHERE user_id = :user_id",
                    },
                ],
                "step_index": 0,
                "step_results": {},
                "past_steps": [],
                "user_id": "user-123",
            }

            result = asyncio.run(EXECUTOR_MODULE.executor_node(state))
        finally:
            EXECUTOR_MODULE.interrupt = original_interrupt

        self.assertIn("Please choose one option", result["step_results"]["s1__pending_input"]["prompt"])
        self.assertEqual(result["step_index"], 0)

    def test_build_execution_summary_includes_rows_for_multi_row_sql_results(self):
        message = EXECUTOR_MODULE.build_execution_summary(
            {
                "past_steps": [
                    {
                        "step_id": "s1",
                        "type": "sql",
                        "input": {},
                        "output": [
                            {"description": "bun cha", "amount": 50000},
                            {"description": "pho", "amount": 60000},
                        ],
                        "status": "success",
                        "summary": "SQL step s1 returned 2 rows",
                    }
                ]
            }
        )

        self.assertIn("bun cha", message.content)
        self.assertIn("pho", message.content)

    def test_build_execution_summary_uses_past_steps_generically(self):
        message = EXECUTOR_MODULE.build_execution_summary(
            {
                "past_steps": [
                    {"step_id": "s1", "type": "tool", "input": {}, "output": {}, "status": "success", "summary": "Selected one option"},
                    {"step_id": "s2", "type": "tool", "input": {}, "output": {}, "status": "error", "summary": "boom"},
                ]
            }
        )

        self.assertIn("Selected one option", message.content)
        self.assertIn("boom", message.content)


if __name__ == "__main__":
    unittest.main()
