import logging
import os
import shlex
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

from .executor import build_execution_summary, executor_node, should_continue_executing
from .planner import clarify_node, planner_node, route_after_planner
from .sql_agent import get_sql_agent
from .state import State
from .tool_registry import register_tools

load_dotenv()
logger = logging.getLogger(__name__)

MAX_MEMORY_MESSAGES = 12
SHORT_TERM_MEMORY: dict[str, InMemoryChatMessageHistory] = {}
MCP_CLIENTS: list[Any] = []


# ---------------------------------------------------------------------------
# Checkpointer — MemorySaver locally, AsyncPostgresSaver in production
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_id(state: State) -> str:
    return str(state.get("session_id") or state.get("user_id") or "default")


def _get_memory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SHORT_TERM_MEMORY:
        SHORT_TERM_MEMORY[session_id] = InMemoryChatMessageHistory()
    return SHORT_TERM_MEMORY[session_id]


def _trim_memory(history: InMemoryChatMessageHistory) -> None:
    if len(history.messages) > MAX_MEMORY_MESSAGES:
        history.messages = history.messages[-MAX_MEMORY_MESSAGES:]


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


def _field_names_from_tool(tool: Any) -> set[str]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return set()
    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    schema_fields = getattr(schema, "__fields__", None)
    if isinstance(schema_fields, dict):
        return set(schema_fields.keys())
    return set()


def _build_sql_agent_messages(state: State) -> list[Any]:
    messages = list(state.get("messages", []))
    runtime_user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()
    if not runtime_user_id:
        return messages

    context = SystemMessage(
        content=(
            "Authenticated runtime context:\n"
            f"- current_user_id: {runtime_user_id}\n"
            "For any query about the current user's data, always filter by this user_id. "
            "Do not ask the user to provide their user_id."
        )
    )
    return [context] + messages


def _escape_prompt_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}").strip()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

async def build_main_graph():
    client = MultiServerMCPClient(_build_mcp_server_config())
    MCP_CLIENTS.append(client)

    tools = await client.get_tools()
    register_tools(tools)                    # make tools available to executor

    tool_node = ToolNode(tools)
    tool_arg_fields = {
        getattr(t, "name", ""): _field_names_from_tool(t) for t in tools
    }
    tools_catalog = "\n".join(
        _escape_prompt_braces(
            f"- {getattr(t, 'name', 'unknown')}: {getattr(t, 'description', '')}"
        )
        for t in tools
    ) or "- No MCP tools available"

    llm = ChatGroq(
        model=os.getenv("ORCHESTRATOR_LLM_MODEL", "openai/gpt-oss-120b"),
        temperature=0.2,
        api_key=os.getenv("groq_api_key"),
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a Vietnamese finance assistant with short-term memory.

You have access to MCP tools:
{tools_catalog}

Rules:
- Use conversation history to resolve references like 'that one' or 'same as before'.
- Call the appropriate MCP tool for structured finance actions.
- Respond naturally in Vietnamese for conversation or incomplete requests.
- Keep replies short and clear.
- ONLY confirm what the user explicitly asked for. Do NOT volunteer summaries,
  totals, or statistics unless the user requests them. Never compute totals
  from memory — those require a database query.
""",
        ),
        MessagesPlaceholder("messages"),
    ])
    llm_with_tools = llm.bind_tools(tools)
    chat_chain = prompt | llm_with_tools

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def chat_node(state: State):
        session_id = _get_session_id(state)
        history = _get_memory(session_id)
        current_messages = list(state.get("messages", []))
        response = await chat_chain.ainvoke({"messages": history.messages + current_messages})
        return {"messages": [response]}

    async def sql_agent_node(state: State):
        messages = _build_sql_agent_messages(state)
        result = await get_sql_agent().ainvoke({"messages": messages})
        final = next(
            (m for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
            AIMessage(content="Mình không tìm được kết quả phù hợp."),
        )
        logger.info("sql_agent_node: %s", str(final.content)[:200])
        return {"messages": [final]}

    def inject_runtime_context_node(state: State):
        current_messages = list(state.get("messages", []))
        last_ai = next(
            (m for m in reversed(current_messages) if isinstance(m, AIMessage)), None
        )
        if last_ai is None or not getattr(last_ai, "tool_calls", None):
            return {}

        runtime_token = str(state.get("token") or os.getenv("TRANSACTION_API_TOKEN", "")).strip()
        runtime_user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()

        updated_tool_calls = []
        changed = False
        for call in last_ai.tool_calls:
            call_copy = dict(call)
            args = dict(call_copy.get("args") or {})
            tool_name = str(call_copy.get("name", ""))
            fields = tool_arg_fields.get(tool_name, set())

            if runtime_token and not args.get("token"):
                args["token"] = runtime_token
                changed = True
            if runtime_user_id:
                if "userId" in fields and not args.get("userId"):
                    args["userId"] = runtime_user_id
                    changed = True
                elif "userId" not in fields and not args.get("user_id"):
                    args["user_id"] = runtime_user_id
                    changed = True

            call_copy["args"] = args
            updated_tool_calls.append(call_copy)

        if not changed:
            return {}

        updated_ai = AIMessage(
            content=last_ai.content,
            additional_kwargs=last_ai.additional_kwargs,
            response_metadata=last_ai.response_metadata,
            id=last_ai.id,
            tool_calls=updated_tool_calls,
            invalid_tool_calls=getattr(last_ai, "invalid_tool_calls", []),
            name=last_ai.name,
        )
        return {"messages": [updated_ai]}

    async def execution_finalize_node(state: State):
        """Finalize for the execute path — builds summary from step results."""
        summary = build_execution_summary(state)
        session_id = _get_session_id(state)
        history = _get_memory(session_id)
        last_user = next(
            (m for m in reversed(list(state.get("messages", [])))
             if isinstance(m, HumanMessage)),
            None,
        )
        if last_user:
            history.add_messages([last_user])
        history.add_messages([summary])
        _trim_memory(history)
        return {"messages": [summary], "response": summary}

    async def finalize_node(state: State):
        session_id = _get_session_id(state)
        history = _get_memory(session_id)
        current_messages = list(state.get("messages", []))

        last_user = next(
            (m for m in reversed(current_messages) if isinstance(m, HumanMessage)), None
        )
        last_ai = next(
            (m for m in reversed(current_messages)
             if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
            None,
        )
        if last_user:
            history.add_messages([last_user])
        if last_ai:
            history.add_messages([last_ai])
            _trim_memory(history)
            return {"response": last_ai}

        fallback = AIMessage(content="Mình chưa có phản hồi phù hợp.")
        history.add_messages([fallback])
        _trim_memory(history)
        return {"response": fallback}

    # ── Graph wiring ──────────────────────────────────────────────────────────

    graph = StateGraph(State)

    graph.add_node("planner",                planner_node)
    graph.add_node("chat",                   chat_node)
    graph.add_node("sql_agent",              sql_agent_node)
    graph.add_node("executor",               executor_node)
    graph.add_node("clarify",                clarify_node)
    graph.add_node("inject_runtime_context", inject_runtime_context_node)
    graph.add_node("tools",                  tool_node)
    graph.add_node("finalize",               finalize_node)
    graph.add_node("execution_finalize",     execution_finalize_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "chat":      "chat",
            "sql_agent": "sql_agent",
            "execute":   "executor",
            "finalize":  "finalize",
            "clarify":   "clarify",
        },
    )

    # chat → MCP tool loop
    graph.add_conditional_edges(
        "chat",
        tools_condition,
        {"tools": "inject_runtime_context", "__end__": "finalize"},
    )
    graph.add_edge("inject_runtime_context", "tools")
    graph.add_edge("tools", "chat")

    # execute → loop → execution_finalize
    graph.add_conditional_edges(
        "executor",
        should_continue_executing,
        {"executor": "executor", "finalize": "execution_finalize"},
    )
    graph.add_edge("execution_finalize", END)

    graph.add_edge("sql_agent", "finalize")
    graph.add_edge("clarify",   "finalize")
    graph.add_edge("finalize",  END)

    return graph.compile(checkpointer=_build_checkpointer())
