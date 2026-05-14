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

_load_module("agents.orchestrator.llm", ROOT / "agents" / "orchestrator" / "llm.py")
_load_module("agents.orchestrator.graph.state", ROOT / "agents" / "orchestrator" / "graph" / "state.py")
PLANNER_MODULE = _load_module(
    "agents.orchestrator.graph.planner",
    ROOT / "agents" / "orchestrator" / "graph" / "planner.py",
)

HumanMessage = PLANNER_MODULE.HumanMessage
planner_node = PLANNER_MODULE.planner_node


class _RaisingLLM:
    async def ainvoke(self, _messages):
        raise RuntimeError(
            "APIError(\"Parsing failed. The model generated output that could not be parsed.\")"
        )


class _StaticLLM:
    def __init__(self, content: str):
        self.content = content

    async def ainvoke(self, _messages):
        return types.SimpleNamespace(content=self.content)


class PlannerFallbackTests(unittest.TestCase):
    def setUp(self):
        self.original_get_llm = PLANNER_MODULE._get_llm

    def tearDown(self):
        PLANNER_MODULE._get_llm = self.original_get_llm

    def test_routes_sql_query_when_classifier_raises_parse_error(self):
        PLANNER_MODULE._get_llm = lambda: _RaisingLLM()

        result = asyncio.run(
            planner_node({"messages": [HumanMessage(content="Tổng chi 1 tháng vừa rồi")]})
        )

        self.assertEqual(result["route"], "sql_agent")

    def test_routes_sql_query_when_classifier_returns_invalid_json(self):
        PLANNER_MODULE._get_llm = lambda: _StaticLLM("not json")

        result = asyncio.run(
            planner_node({"messages": [HumanMessage(content="Tổng chi 1 tháng vừa rồi")]})
        )

        self.assertEqual(result["route"], "sql_agent")


if __name__ == "__main__":
    unittest.main()
