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
_load_module("agents.orchestrator.graph.planner", ROOT / "agents" / "orchestrator" / "graph" / "planner.py")
_load_module("agents.orchestrator.graph.sql_logging", ROOT / "agents" / "orchestrator" / "graph" / "sql_logging.py")
_load_module("agents.orchestrator.graph.sql_agent", ROOT / "agents" / "orchestrator" / "graph" / "sql_agent.py")
MAIN_GRAPH_MODULE = _load_module(
    "agents.orchestrator.graph.main_graph",
    ROOT / "agents" / "orchestrator" / "graph" / "main_graph.py",
)


class MainGraphDirectSqlRemovedTests(unittest.TestCase):
    def test_main_graph_does_not_expose_direct_sql_report_helper(self):
        self.assertFalse(hasattr(MAIN_GRAPH_MODULE, "_try_direct_sql_report"))


if __name__ == "__main__":
    unittest.main()
