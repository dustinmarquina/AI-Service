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
_load_module("agents.orchestrator.graph.planner", ROOT / "agents" / "orchestrator" / "graph" / "planner.py")
_load_module("agents.orchestrator.graph.sql_logging", ROOT / "agents" / "orchestrator" / "graph" / "sql_logging.py")
_load_module("agents.orchestrator.graph.sql_agent", ROOT / "agents" / "orchestrator" / "graph" / "sql_agent.py")
MAIN_GRAPH_MODULE = _load_module(
    "agents.orchestrator.graph.main_graph",
    ROOT / "agents" / "orchestrator" / "graph" / "main_graph.py",
)


class _ExplodingClient:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("MultiServerMCPClient should not be created during graph build")


class _ExplodingChatGroq:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("ChatGroq should not be created during graph build")


class MainGraphLazyInitTests(unittest.TestCase):
    def setUp(self):
        self.original_client = MAIN_GRAPH_MODULE.MultiServerMCPClient
        self.original_chatgroq = MAIN_GRAPH_MODULE.ChatGroq
        self.original_graph = getattr(MAIN_GRAPH_MODULE, "_main_graph", None)
        self.original_runtime = getattr(MAIN_GRAPH_MODULE, "_chat_runtime", None)
        self.original_lock = getattr(MAIN_GRAPH_MODULE, "_chat_runtime_lock", None)

        MAIN_GRAPH_MODULE._main_graph = None
        MAIN_GRAPH_MODULE._chat_runtime = None
        MAIN_GRAPH_MODULE._chat_runtime_lock = None

    def tearDown(self):
        MAIN_GRAPH_MODULE.MultiServerMCPClient = self.original_client
        MAIN_GRAPH_MODULE.ChatGroq = self.original_chatgroq
        MAIN_GRAPH_MODULE._main_graph = self.original_graph
        MAIN_GRAPH_MODULE._chat_runtime = self.original_runtime
        MAIN_GRAPH_MODULE._chat_runtime_lock = self.original_lock

    def test_build_main_graph_does_not_initialize_expensive_runtime(self):
        MAIN_GRAPH_MODULE.MultiServerMCPClient = _ExplodingClient
        MAIN_GRAPH_MODULE.ChatGroq = _ExplodingChatGroq

        graph = asyncio.run(MAIN_GRAPH_MODULE.build_main_graph())

        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
