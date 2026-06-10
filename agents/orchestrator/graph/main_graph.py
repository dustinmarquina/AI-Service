import logging
import os
import shlex
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient

from .executor import build_execution_summary, executor_node
from .planner import clarify_node, planner_node, route_after_planner
from .replanner import replanner_node, route_after_replanner
from .state import State
from .tool_registry import register_tools

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MAX_REPLAN_ATTEMPTS = int(os.getenv("ORCHESTRATOR_MAX_REPLAN_ATTEMPTS", "2"))
MCP_CLIENTS: list[Any] = []


def _build_checkpointer():
    postgres_uri = os.getenv("POSTGRES_URI", "").strip()
    use_pg = os.getenv("USE_POSTGRES_CHECKPOINTER", "false").lower() == "true"

    if postgres_uri and use_pg:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            checkpointer = AsyncPostgresSaver.from_conn_string(postgres_uri)
            logger.info("checkpointer: using AsyncPostgresSaver")
            return checkpointer
        except ImportError:
            logger.warning(
                "checkpointer: langgraph-checkpoint-postgres not installed, "
                "falling back to MemorySaver"
            )

    logger.info("checkpointer: using MemorySaver")
    return MemorySaver()


def _build_mcp_server_config() -> dict[str, dict[str, Any]]:
    command = os.getenv("ORCHESTRATOR_MCP_COMMAND", "python")
    args = os.getenv("ORCHESTRATOR_MCP_ARGS", "-m agents.orchestrator.mcp_server")
    return {
        "orchestrator": {
            "transport": os.getenv("ORCHESTRATOR_MCP_TRANSPORT", "stdio"),
            "command": command,
            "args": shlex.split(args),
        }
    }


def _ensure_replan_defaults(state: State) -> dict[str, Any]:
    update: dict[str, Any] = {}
    if state.get("past_steps") is None:
        update["past_steps"] = []
    if state.get("replan_attempts") is None:
        update["replan_attempts"] = 0
    if state.get("max_replan_attempts") is None:
        update["max_replan_attempts"] = DEFAULT_MAX_REPLAN_ATTEMPTS
    return update


def _build_finalize_response(state: State) -> dict[str, Any]:
    if state.get("past_steps"):
        summary = build_execution_summary(state)
        return {"messages": [summary], "response": summary}

    current_messages = list(state.get("messages", []))
    last_ai = next(
        (
            message for message in reversed(current_messages)
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
        ),
        None,
    )
    if last_ai is not None:
        return {"response": last_ai}

    summary = build_execution_summary(state)
    return {"messages": [summary], "response": summary}


async def build_main_graph():
    client = MultiServerMCPClient(_build_mcp_server_config())
    MCP_CLIENTS.append(client)
    tools = await client.get_tools()
    register_tools(tools)

    async def initialize_node(state: State):
        return _ensure_replan_defaults(state)

    async def finalize_node(state: State):
        return _build_finalize_response(state)

    graph = StateGraph(State)
    graph.add_node("initialize", initialize_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "execute": "executor",
            "finalize": "finalize",
            "clarify": "clarify",
        },
    )
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "executor": "executor",
            "finalize": "finalize",
            "clarify": "clarify",
        },
    )
    graph.add_edge("clarify", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=_build_checkpointer())
