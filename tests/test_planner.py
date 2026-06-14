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
        self.tool_calls = []


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
sys.modules["agents.orchestrator.llm"] = types.SimpleNamespace(get_classifier_llm=lambda: None)
sys.modules["agents.orchestrator.graph.tool_registry"] = types.SimpleNamespace(describe_tools=lambda: "- TOOL demo_tool()")

_ensure_package("agents", ROOT / "agents")
_ensure_package("agents.orchestrator", ROOT / "agents" / "orchestrator")
_ensure_package("agents.orchestrator.graph", ROOT / "agents" / "orchestrator" / "graph")

_load_module("agents.orchestrator.graph.state", ROOT / "agents" / "orchestrator" / "graph" / "state.py")
PLANNER_MODULE = _load_module(
    "agents.orchestrator.graph.planner",
    ROOT / "agents" / "orchestrator" / "graph" / "planner.py",
)


class _StaticLLM:
    def __init__(self, content: str):
        self.content = content

    async def ainvoke(self, _messages):
        return types.SimpleNamespace(content=self.content)


HumanMessage = PLANNER_MODULE.HumanMessage
planner_node = PLANNER_MODULE.planner_node


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.original_get_llm = PLANNER_MODULE._get_llm

    def tearDown(self):
        PLANNER_MODULE._get_llm = self.original_get_llm

    def test_execute_route_initializes_step_state(self):
        PLANNER_MODULE._get_llm = lambda: _StaticLLM(
            '{"route":"execute","reason":"need tool","steps":[{"id":"s1","type":"tool","name":"demo_tool","reasoning":"Need to call the demo tool","args":{}}]}'
        )

        result = asyncio.run(planner_node({"messages": [HumanMessage(content="do something")]}))

        self.assertEqual(result["route"], "execute")
        self.assertEqual(result["step_index"], 0)
        self.assertEqual(result["step_results"], {})
        self.assertEqual(result["past_steps"], [])
        self.assertEqual(result["replan_attempts"], 0)
        self.assertEqual(result["steps"][0]["name"], "get_user_id")
        self.assertEqual(result["steps"][1]["reasoning"], "Need to call the demo tool")

    def test_invalid_json_falls_back_to_clarify(self):
        PLANNER_MODULE._get_llm = lambda: _StaticLLM("not json")

        result = asyncio.run(planner_node({"messages": [HumanMessage(content="do something")]}))

        self.assertEqual(result["route"], "clarify")
        self.assertIn("more information", result["messages"][0].content.lower())

    def test_finalize_route_returns_message(self):
        PLANNER_MODULE._get_llm = lambda: _StaticLLM(
            '{"route":"finalize","reason":"simple answer","message":"Done."}'
        )

        result = asyncio.run(planner_node({"messages": [HumanMessage(content="hello")]}))

        self.assertEqual(result["route"], "finalize")
        self.assertEqual(result["messages"][0].content, "Done.")

    def test_route_after_planner_rejects_removed_sql_agent_route(self):
        route = PLANNER_MODULE.route_after_planner({"route": "sql_agent"})
        self.assertEqual(route, "clarify")

    def test_prompt_is_domain_agnostic(self):
        prompt = PLANNER_MODULE._build_planner_system_prompt()
        self.assertNotIn("finance assistant", prompt.lower())
        self.assertIn('Do not use "clarify" just because an optional tool argument is missing.', prompt)
        self.assertIn('prefer "execute" over "clarify"', prompt)

    def test_prompt_is_built_from_live_tool_descriptions(self):
        original_describe_tools = PLANNER_MODULE.describe_tools
        try:
            PLANNER_MODULE.describe_tools = lambda: "- TOOL dynamic_tool(input [required])"
            prompt = PLANNER_MODULE._build_planner_system_prompt()
        finally:
            PLANNER_MODULE.describe_tools = original_describe_tools

        self.assertIn("dynamic_tool", prompt)

    def test_execute_route_requires_reasoning_per_step(self):
        PLANNER_MODULE._get_llm = lambda: _StaticLLM(
            '{"route":"execute","reason":"need tool","steps":[{"id":"s1","type":"tool","name":"demo_tool","args":{}}]}'
        )

        result = asyncio.run(planner_node({"messages": [HumanMessage(content="do something")]}))

        self.assertEqual(result["route"], "clarify")

    def test_nested_placeholder_syntax_is_rejected(self):
        PLANNER_MODULE._get_llm = lambda: _StaticLLM(
            '{"route":"execute","reason":"need lookup","steps":[{"id":"s1","type":"sql","reasoning":"Need to fetch wallet rows first","query_hint":"SELECT id, name FROM wallets"},{"id":"s2","type":"tool","name":"demo_tool","reasoning":"Need wallet id for the tool call","args":{"wallet_id":"$s1.wallets[0].walletId"}}]}'
        )

        result = asyncio.run(planner_node({"messages": [HumanMessage(content="do something")]}))

        self.assertEqual(result["route"], "clarify")

    def test_clarify_transaction_entry_falls_back_to_execute(self):
        original_describe_tools = PLANNER_MODULE.describe_tools
        try:
            PLANNER_MODULE.describe_tools = lambda: "- TOOL create_transaction(amount: str [required], description: str [required], wallet_name: str [optional])"
            PLANNER_MODULE._get_llm = lambda: _StaticLLM(
                '{"route":"clarify","reason":"wallet unknown","message":"Bạn muốn ghi chi tiêu 150k cho cà ri Ấn Độ vào ví nào?"}'
            )

            result = asyncio.run(planner_node({"messages": [HumanMessage(content="150k cà ri Ấn Độ vào ví MB Bank")]}))
        finally:
            PLANNER_MODULE.describe_tools = original_describe_tools

        self.assertEqual(result["route"], "execute")
        self.assertEqual(result["steps"][1]["type"], "sql")
        self.assertEqual(result["steps"][1]["selection_mode"], "none")
        self.assertIn("FROM wallets", result["steps"][1]["query_hint"])
        self.assertEqual(result["steps"][2]["name"], "create_transaction")
        self.assertEqual(result["steps"][2]["args"]["amount"], "150k")
        self.assertNotIn("wallet_name", result["steps"][2]["args"])
        self.assertEqual(result["steps"][2]["args"]["description"], "cà ri Ấn Độ")

    def test_generic_transaction_command_is_not_used_as_description(self):
        original_describe_tools = PLANNER_MODULE.describe_tools
        try:
            PLANNER_MODULE.describe_tools = lambda: "- TOOL create_transaction(amount: str [required], description: str [required], wallet_name: str [optional])"
            PLANNER_MODULE._get_llm = lambda: _StaticLLM(
                '{"route":"clarify","reason":"need description","message":"Bạn muốn thêm giao dịch gì?"}'
            )

            result = asyncio.run(planner_node({"messages": [HumanMessage(content="thêm giao dịch 51k")]}))
        finally:
            PLANNER_MODULE.describe_tools = original_describe_tools

        self.assertEqual(result["route"], "clarify")
        self.assertNotIn("steps", result)

    def test_dependency_lookup_with_order_by_limit_is_rewritten_to_candidate_query(self):
        PLANNER_MODULE._get_llm = lambda: _StaticLLM(
            '{"route":"execute","reason":"need wallet first","steps":[{"id":"s1","type":"sql","reasoning":"Select the wallet with the lowest balance for this user so the transaction can be recorded there.","query_hint":"SELECT id FROM wallets WHERE user_id = $s0.user_id ORDER BY balance ASC LIMIT 1","selection_mode":"required"},{"id":"s2","type":"tool","name":"create_transaction","reasoning":"Create the expense transaction in the chosen wallet.","args":{"amount":"50k","description":"nem chua Hà Nội","wallet_id":"$s1.id"}}]}'
        )

        result = asyncio.run(planner_node({"messages": [HumanMessage(content="50k nem chua Hà Nội vào ví có số dư thấp nhất")]}))

        self.assertEqual(result["route"], "execute")
        self.assertEqual(result["steps"][1]["type"], "sql")
        self.assertEqual(result["steps"][1]["selection_mode"], "none")
        self.assertEqual(
            result["steps"][1]["query_hint"],
            "SELECT id, name, balance, currency FROM wallets WHERE user_id = $s0.user_id",
        )
        self.assertNotIn("wallet_id", result["steps"][2]["args"])


if __name__ == "__main__":
    unittest.main()
