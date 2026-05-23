import importlib.util
import sys
import types
import unittest
from pathlib import Path

from langchain_core.messages import SystemMessage


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


class MainGraphSqlAgentContextTests(unittest.TestCase):
    def test_build_sql_agent_messages_includes_authenticated_user_id(self):
        messages = [HumanMessage(content="lần chi tiêu gần nhất")]

        sql_messages = MAIN_GRAPH_MODULE._build_sql_agent_messages(
            {"messages": messages, "user_id": "af1eb20b-54e5-4910-b974-60151185e48f"}
        )

        self.assertIsInstance(sql_messages[0], SystemMessage)
        self.assertIn("af1eb20b-54e5-4910-b974-60151185e48f", sql_messages[0].content)
        self.assertIn("Do not ask the user to provide their user_id", sql_messages[0].content)
        self.assertEqual(sql_messages[1:], messages)


if __name__ == "__main__":
    unittest.main()
