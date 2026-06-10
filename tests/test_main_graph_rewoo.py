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


sys.modules["langchain_core.messages"] = types.SimpleNamespace(
    AIMessage=_FakeAIMessage,
    BaseMessage=object,
    HumanMessage=_FakeHumanMessage,
)
sys.modules["langgraph.graph.message"] = types.SimpleNamespace(add_messages=lambda x, y: x + y)
sys.modules["langgraph.checkpoint.memory"] = types.SimpleNamespace(MemorySaver=lambda: object())
sys.modules["langgraph.graph"] = types.SimpleNamespace(END="END", START="START", StateGraph=object)
sys.modules["langchain_mcp_adapters.client"] = types.SimpleNamespace(MultiServerMCPClient=object)
sys.modules["agents.orchestrator.graph.executor"] = types.SimpleNamespace(
    build_execution_summary=lambda _state: _FakeAIMessage(content="summary"),
    executor_node=lambda _state: {},
)
sys.modules["agents.orchestrator.graph.planner"] = types.SimpleNamespace(
    clarify_node=lambda _state: {},
    planner_node=lambda _state: {},
    route_after_planner=lambda _state: "clarify",
)
sys.modules["agents.orchestrator.graph.replanner"] = types.SimpleNamespace(
    replanner_node=lambda _state: {},
    route_after_replanner=lambda _state: "clarify",
)
sys.modules["agents.orchestrator.graph.tool_registry"] = types.SimpleNamespace(register_tools=lambda _tools: None)

_ensure_package("agents", ROOT / "agents")
_ensure_package("agents.orchestrator", ROOT / "agents" / "orchestrator")
_ensure_package("agents.orchestrator.graph", ROOT / "agents" / "orchestrator" / "graph")

_load_module("agents.orchestrator.graph.state", ROOT / "agents" / "orchestrator" / "graph" / "state.py")
MAIN_GRAPH_MODULE = _load_module(
    "agents.orchestrator.graph.main_graph",
    ROOT / "agents" / "orchestrator" / "graph" / "main_graph.py",
)


class MainGraphRewooTests(unittest.TestCase):
    def test_replan_defaults_are_initialized(self):
        update = MAIN_GRAPH_MODULE._ensure_replan_defaults({})

        self.assertEqual(update["past_steps"], [])
        self.assertEqual(update["replan_attempts"], 0)
        self.assertEqual(update["max_replan_attempts"], MAIN_GRAPH_MODULE.DEFAULT_MAX_REPLAN_ATTEMPTS)

    def test_finalize_prefers_fresh_summary_over_stale_ai_message_when_past_steps_exist(self):
        stale = _FakeAIMessage(content="old summary")
        result = MAIN_GRAPH_MODULE._build_finalize_response(
            {
                "route": "finalize",
                "messages": [_FakeHumanMessage("50k"), stale],
                "past_steps": [{"step_id": "s1", "status": "success", "summary": "create_transaction completed"}],
            }
        )

        self.assertEqual(result["response"].content, "summary")
        self.assertEqual(result["messages"][0].content, "summary")


if __name__ == "__main__":
    unittest.main()
