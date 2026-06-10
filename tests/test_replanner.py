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


class _FakeAIMessage:
    def __init__(self, content="", **_kwargs):
        self.content = content


class _FakeHumanMessage:
    def __init__(self, content="", **_kwargs):
        self.content = content


class _FakeSystemMessage:
    def __init__(self, content="", **_kwargs):
        self.content = content


sys.modules["langchain_core.messages"] = types.SimpleNamespace(
    AIMessage=_FakeAIMessage,
    BaseMessage=object,
    HumanMessage=_FakeHumanMessage,
    SystemMessage=_FakeSystemMessage,
)
sys.modules["langgraph.graph.message"] = types.SimpleNamespace(add_messages=lambda x, y: x + y)

_ensure_package("agents", ROOT / "agents")
_ensure_package("agents.orchestrator", ROOT / "agents" / "orchestrator")
_ensure_package("agents.orchestrator.graph", ROOT / "agents" / "orchestrator" / "graph")

sys.modules["agents.orchestrator.llm"] = types.SimpleNamespace(get_classifier_llm=lambda: None)

_load_module("agents.orchestrator.graph.state", ROOT / "agents" / "orchestrator" / "graph" / "state.py")
REPLANNER_MODULE = _load_module(
    "agents.orchestrator.graph.replanner",
    ROOT / "agents" / "orchestrator" / "graph" / "replanner.py",
)


class _StaticLLM:
    def __init__(self, content: str):
        self.content = content

    async def ainvoke(self, _messages):
        return types.SimpleNamespace(content=self.content)


class _RecordingLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return types.SimpleNamespace(content=self.content)


class ReplannerTests(unittest.TestCase):
    def setUp(self):
        self.original_get_llm = REPLANNER_MODULE._get_llm

    def tearDown(self):
        REPLANNER_MODULE._get_llm = self.original_get_llm

    def test_replanner_increments_attempts_when_emitting_new_steps(self):
        REPLANNER_MODULE._get_llm = lambda: _StaticLLM(
            '{"route":"execute","reason":"retry with new steps","steps":[{"id":"s2","type":"tool","name":"retry_tool","args":{}}]}'
        )

        result = asyncio.run(
            REPLANNER_MODULE.replanner_node(
                {
                    "messages": [_FakeHumanMessage(content="do something")],
                    "steps": [{"id": "s1", "type": "tool", "name": "broken_tool", "args": {}}],
                    "step_index": 1,
                    "past_steps": [{"step_id": "s1", "type": "tool", "input": {}, "output": {}, "status": "error", "summary": "boom"}],
                    "replan_attempts": 0,
                    "max_replan_attempts": 2,
                }
            )
        )

        self.assertEqual(result["route"], "execute")
        self.assertEqual(result["replan_attempts"], 1)
        self.assertEqual(result["steps"][0]["name"], "retry_tool")

    def test_replanner_stops_when_attempt_limit_reached(self):
        result = asyncio.run(
            REPLANNER_MODULE.replanner_node(
                {
                    "messages": [_FakeHumanMessage(content="do something")],
                    "steps": [{"id": "s1", "type": "tool", "name": "broken_tool", "args": {}}],
                    "step_index": 1,
                    "past_steps": [{"step_id": "s1", "type": "tool", "input": {}, "output": {}, "status": "error", "summary": "boom"}],
                    "replan_attempts": 2,
                    "max_replan_attempts": 2,
                }
            )
        )

        self.assertEqual(result["route"], "clarify")
        self.assertIn("boom", result["messages"][0].content)

    def test_route_after_replanner_returns_executor_for_execute(self):
        route = REPLANNER_MODULE.route_after_replanner({"route": "execute"})
        self.assertEqual(route, "executor")

    def test_replanner_clarifies_after_multiple_failed_cycles(self):
        state = {
            "messages": [_FakeHumanMessage(content="do something")],
            "steps": [{"id": "s1", "type": "tool", "name": "broken_tool", "args": {}}],
            "step_index": 1,
            "past_steps": [{"step_id": "s1", "type": "tool", "input": {}, "output": {}, "status": "error", "summary": "boom"}],
            "max_replan_attempts": 2,
        }

        first = asyncio.run(REPLANNER_MODULE.replanner_node({**state, "replan_attempts": 2}))
        second = asyncio.run(REPLANNER_MODULE.replanner_node({**state, "replan_attempts": 5}))

        self.assertEqual(first["route"], "clarify")
        self.assertEqual(second["route"], "clarify")

    def test_replanner_binds_missing_id_arg_from_prior_sql_rows(self):
        llm = _RecordingLLM(
            '{"route":"execute","reason":"MB Bank is a direct match among the candidate accounts.","bindings":{"account_id":"account-2"}}'
        )
        REPLANNER_MODULE._get_llm = lambda: llm

        result = asyncio.run(
            REPLANNER_MODULE.replanner_node(
                {
                    "messages": [_FakeHumanMessage(content="pay this invoice with MB Bank")],
                    "steps": [
                        {"id": "s0", "type": "tool", "name": "get_user_id", "args": {}},
                        {
                            "id": "s1",
                            "type": "sql",
                            "query_hint": "SELECT id, name, provider FROM accounts WHERE user_id = $s0.user_id AND active = true",
                            "selection_mode": "none",
                        },
                        {
                            "id": "s2",
                            "type": "tool",
                            "name": "create_payment",
                            "reasoning": "create_payment requires account_id, which is not yet known, so bind it from prior SQL results if the user text identifies one account.",
                            "args": {"amount": "50k", "description": "invoice"},
                        },
                    ],
                    "step_index": 2,
                    "step_results": {
                        "s0": {"status": "success", "user_id": "user-123"},
                        "s1": [
                            {"id": "account-1", "name": "Cash", "provider": "Local"},
                            {"id": "account-2", "name": "MB Bank", "provider": "MB"},
                        ],
                    },
                    "past_steps": [
                        {"step_id": "s0", "type": "tool", "input": {}, "output": {"status": "success", "user_id": "user-123"}, "status": "success", "summary": "get_user_id completed"},
                        {"step_id": "s1", "type": "sql", "input": {}, "output": [{"id": "account-1", "name": "Cash"}, {"id": "account-2", "name": "MB Bank"}], "status": "success", "summary": "SQL step s1 returned 2 rows"},
                    ],
                    "replan_attempts": 0,
                    "max_replan_attempts": 2,
                }
            )
        )

        self.assertEqual(result["route"], "execute")
        self.assertEqual(result["steps"][2]["args"]["account_id"], "account-2")
        self.assertEqual(result["step_index"], 2)
        self.assertEqual(len(llm.calls), 1)

    def test_replanner_binds_missing_id_arg_from_lowest_balance_candidate(self):
        llm = _RecordingLLM(
            '{"route":"execute","reason":"The user asked for the wallet with the lowest balance, which is wallet-3.","bindings":{"wallet_id":"wallet-3"}}'
        )
        REPLANNER_MODULE._get_llm = lambda: llm

        result = asyncio.run(
            REPLANNER_MODULE.replanner_node(
                {
                    "messages": [_FakeHumanMessage(content="50k bún bò huế vào ví có số dư thấp nhất")],
                    "steps": [
                        {"id": "s0", "type": "tool", "name": "get_user_id", "args": {}},
                        {
                            "id": "s1",
                            "type": "sql",
                            "reasoning": "create_transaction requires wallet_id, so fetch wallet candidates with their balances first.",
                            "query_hint": "SELECT id, name, balance, currency FROM wallets WHERE user_id = $s0.user_id AND active = true",
                            "selection_mode": "none",
                        },
                        {
                            "id": "s2",
                            "type": "tool",
                            "name": "create_transaction",
                            "reasoning": "create_transaction requires wallet_id, so bind it from the prior wallet candidates before writing the transaction.",
                            "args": {"amount": "50k", "description": "bún bò huế"},
                        },
                    ],
                    "step_index": 2,
                    "step_results": {
                        "s0": {"status": "success", "user_id": "user-123"},
                        "s1": [
                            {"id": "wallet-1", "name": "Du học", "balance": 400000000, "currency": "VND"},
                            {"id": "wallet-2", "name": "MB Bank", "balance": 50000000, "currency": "VND"},
                            {"id": "wallet-3", "name": "Cash", "balance": 250000, "currency": "VND"},
                        ],
                    },
                    "past_steps": [
                        {"step_id": "s0", "type": "tool", "input": {}, "output": {"status": "success", "user_id": "user-123"}, "status": "success", "summary": "get_user_id completed"},
                        {"step_id": "s1", "type": "sql", "input": {}, "output": [{"id": "wallet-1", "name": "Du học"}, {"id": "wallet-2", "name": "MB Bank"}, {"id": "wallet-3", "name": "Cash"}], "status": "success", "summary": "SQL step s1 returned 3 rows"},
                    ],
                    "replan_attempts": 0,
                    "max_replan_attempts": 2,
                }
            )
        )

        self.assertEqual(result["route"], "execute")
        self.assertEqual(result["steps"][2]["args"]["wallet_id"], "wallet-3")
        self.assertEqual(result["step_index"], 2)
        self.assertEqual(len(llm.calls), 0)

    def test_replanner_auto_binds_single_candidate_without_llm(self):
        llm = _RecordingLLM(
            '{"route":"execute","reason":"should not be used","bindings":{"wallet_id":"wallet-1"}}'
        )
        REPLANNER_MODULE._get_llm = lambda: llm

        result = asyncio.run(
            REPLANNER_MODULE.replanner_node(
                {
                    "messages": [_FakeHumanMessage(content="ghi 50k bún chả")],
                    "steps": [
                        {"id": "s0", "type": "tool", "name": "get_user_id", "args": {}},
                        {
                            "id": "s1",
                            "type": "sql",
                            "query_hint": "SELECT id, name, balance FROM wallets WHERE user_id = $s0.user_id AND active = true",
                            "selection_mode": "none",
                        },
                        {
                            "id": "s2",
                            "type": "tool",
                            "name": "create_transaction",
                            "reasoning": "create_transaction requires wallet_id before execution.",
                            "args": {"amount": "50k", "description": "bún chả"},
                        },
                    ],
                    "step_index": 2,
                    "step_results": {
                        "s0": {"status": "success", "user_id": "user-123"},
                        "s1": [
                            {"id": "wallet-1", "name": "Cash", "balance": 250000},
                        ],
                    },
                    "past_steps": [
                        {"step_id": "s0", "type": "tool", "input": {}, "output": {"status": "success", "user_id": "user-123"}, "status": "success", "summary": "get_user_id completed"},
                        {"step_id": "s1", "type": "sql", "input": {}, "output": [{"id": "wallet-1", "name": "Cash"}], "status": "success", "summary": "SQL step s1 returned 1 row"},
                    ],
                    "replan_attempts": 0,
                    "max_replan_attempts": 2,
                }
            )
        )

        self.assertEqual(result["route"], "execute")
        self.assertEqual(result["steps"][2]["args"]["wallet_id"], "wallet-1")
        self.assertEqual(len(llm.calls), 0)

    def test_replanner_rejects_invalid_llm_binding_and_falls_through(self):
        llm = _RecordingLLM(
            '{"route":"execute","reason":"picked an invalid row","bindings":{"wallet_id":"wallet-999"}}'
        )
        REPLANNER_MODULE._get_llm = lambda: llm

        result = asyncio.run(
            REPLANNER_MODULE.replanner_node(
                {
                    "messages": [_FakeHumanMessage(content="50k bún bò huế vào ví phù hợp")],
                    "steps": [
                        {"id": "s0", "type": "tool", "name": "get_user_id", "args": {}},
                        {
                            "id": "s1",
                            "type": "sql",
                            "reasoning": "create_transaction requires wallet_id, so fetch wallet candidates first.",
                            "query_hint": "SELECT id, name, balance FROM wallets WHERE user_id = $s0.user_id AND active = true",
                            "selection_mode": "none",
                        },
                        {
                            "id": "s2",
                            "type": "tool",
                            "name": "create_transaction",
                            "reasoning": "create_transaction requires wallet_id before execution.",
                            "args": {"amount": "50k", "description": "bún bò huế"},
                        },
                    ],
                    "step_index": 2,
                    "step_results": {
                        "s0": {"status": "success", "user_id": "user-123"},
                        "s1": [
                            {"id": "wallet-1", "name": "Du học", "balance": 400000000},
                            {"id": "wallet-2", "name": "MB Bank", "balance": 50000000},
                        ],
                    },
                    "past_steps": [
                        {"step_id": "s0", "type": "tool", "input": {}, "output": {"status": "success", "user_id": "user-123"}, "status": "success", "summary": "get_user_id completed"},
                        {"step_id": "s1", "type": "sql", "input": {}, "output": [{"id": "wallet-1", "name": "Du học"}, {"id": "wallet-2", "name": "MB Bank"}], "status": "success", "summary": "SQL step s1 returned 2 rows"},
                    ],
                    "replan_attempts": 0,
                    "max_replan_attempts": 2,
                }
            )
        )

        self.assertEqual(result["route"], "execute")
        self.assertNotIn("steps", result)
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
