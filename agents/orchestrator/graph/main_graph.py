import os
import shlex
import asyncio
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
import logging

from .planner import clarify_node, planner_node, route_after_planner
from .sql_agent import get_sql_agent
from .sql_logging import extract_faulty_sql_from_messages
from .state import State

load_dotenv()

logger = logging.getLogger(__name__)

MAX_MEMORY_MESSAGES = 12
SHORT_TERM_MEMORY: dict[str, InMemoryChatMessageHistory] = {}
MCP_CLIENTS: list[Any] = []
_main_graph = None
_main_graph_lock: asyncio.Lock | None = None
_chat_runtime = None
_chat_runtime_lock: asyncio.Lock | None = None


@dataclass
class ChatRuntime:
    tool_node: ToolNode
    tool_arg_fields: dict[str, set[str]]
    chat_chain: Any


# ---------------------------------------------------------------------------
# Helpers (unchanged)
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


async def _build_chat_runtime() -> ChatRuntime:
    client = MultiServerMCPClient(_build_mcp_server_config())
    MCP_CLIENTS.append(client)

    tools = await client.get_tools()
    tool_node = ToolNode(tools)
    tool_arg_fields = {
        getattr(tool, "name", ""): _field_names_from_tool(tool)
        for tool in tools
    }

    tools_catalog = "\n".join(
        f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '')}".strip()
        for tool in tools
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

You have access to MCP tools discovered at runtime:
{tools_catalog}

Rules:
- Use the conversation history to resolve references like 'that one', 'same as before', or omitted details.
- Call the appropriate MCP tool whenever the user wants a structured finance action.
- If the user is just chatting or the request is incomplete, respond naturally in Vietnamese.
- Keep replies short and clear.
- ONLY confirm what the user explicitly asked for. Do NOT volunteer summaries,
  totals, or statistics unless the user requests them. Never compute totals
  from memory — those require a database query.
""",
        ),
        MessagesPlaceholder("messages"),
    ])

    return ChatRuntime(
        tool_node=tool_node,
        tool_arg_fields=tool_arg_fields,
        chat_chain=prompt | llm.bind_tools(tools),
    )


async def _get_chat_runtime() -> ChatRuntime:
    global _chat_runtime
    global _chat_runtime_lock

    if _chat_runtime is not None:
        return _chat_runtime

    if _chat_runtime_lock is None:
        _chat_runtime_lock = asyncio.Lock()

    async with _chat_runtime_lock:
        if _chat_runtime is None:
            _chat_runtime = await _build_chat_runtime()

    return _chat_runtime


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

async def build_main_graph():
    global _main_graph
    global _main_graph_lock

    if _main_graph is not None:
        return _main_graph

    if _main_graph_lock is None:
        _main_graph_lock = asyncio.Lock()

    async with _main_graph_lock:
        if _main_graph is not None:
            return _main_graph

        # ---------------------------------------------------------------------------
        # Nodes
        # ---------------------------------------------------------------------------

        async def chat_node(state: State):
            runtime = await _get_chat_runtime()
            session_id = _get_session_id(state)
            history = _get_memory(session_id)
            current_messages = list(state.get("messages", []))
            prompt_messages = history.messages + current_messages
            response = await runtime.chat_chain.ainvoke({"messages": prompt_messages})
            return {"messages": [response]}

        async def sql_agent_node(state: State):
            """Delegate to the SQL subgraph and surface its final answer."""
            messages = list(state.get("messages", []))

            # Inject runtime user_id (and token if present) into the SQL agent context
            runtime_user_id = str(state.get("user_id") or "").strip()
            runtime_token = str(state.get("token") or "").strip()
            context_msgs = []
            if runtime_user_id:
                context_msgs.append(SystemMessage(content=f"RuntimeContext: user_id={runtime_user_id}"))
            if runtime_token:
                context_msgs.append(SystemMessage(content=f"RuntimeContext: token={runtime_token}"))

            sql_agent = get_sql_agent()
            result = await sql_agent.ainvoke({"messages": context_msgs + messages})
            faulty_sql = extract_faulty_sql_from_messages(result.get("messages", []))

            # Extract the last non-tool AIMessage as the answer
            final = next(
                (
                    m for m in reversed(result["messages"])
                    if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
                ),
                AIMessage(content="Mình không tìm được kết quả phù hợp."),
            )
            if faulty_sql:
                logger.warning("sql_agent_node: faulty_sql=%s", faulty_sql)
            logger.info("sql_agent_node: answer=%s", str(final.content)[:200])
            return {"messages": [final]}

        async def inject_runtime_context_node(state: State):
            runtime = await _get_chat_runtime()
            current_messages = list(state.get("messages", []))
            if not current_messages:
                return {}

            last_ai = next(
                (m for m in reversed(current_messages) if isinstance(m, AIMessage)),
                None,
            )
            if last_ai is None or not getattr(last_ai, "tool_calls", None):
                return {}

            runtime_token = str(state.get("token") or os.getenv("TRANSACTION_API_TOKEN", "")).strip()
            runtime_user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()

            redacted = (runtime_token[:8] + "...") if runtime_token else "(none)"
            logger.info("inject_runtime_context: token=%s user_id=%s", redacted, runtime_user_id or "(none)")

            updated_tool_calls = []
            changed = False

            for call in last_ai.tool_calls:
                call_copy = dict(call)
                args = dict(call_copy.get("args") or {})
                tool_name = str(call_copy.get("name", ""))
                fields = runtime.tool_arg_fields.get(tool_name, set())

                if runtime_token and not args.get("token"):
                    args["token"] = runtime_token
                    changed = True
                    logger.info("inject_runtime_context: injected 'token' into '%s'", tool_name)

                if runtime_user_id:
                    if "userId" in fields:
                        if not args.get("userId"):
                            args["userId"] = runtime_user_id
                            changed = True
                    else:
                        if not args.get("user_id"):
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

        async def tools_node(state: State):
            runtime = await _get_chat_runtime()
            return await runtime.tool_node.ainvoke(state)

        async def finalize_node(state: State):
            session_id = _get_session_id(state)
            history = _get_memory(session_id)
            current_messages = list(state.get("messages", []))

            last_user_message = next(
                (m for m in reversed(current_messages) if isinstance(m, HumanMessage)), None
            )
            last_ai_message = next(
                (
                    m for m in reversed(current_messages)
                    if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
                ),
                None,
            )

            if last_user_message:
                history.add_messages([last_user_message])

            if last_ai_message:
                history.add_messages([last_ai_message])
                _trim_memory(history)
                return {"response": last_ai_message}

            fallback = AIMessage(content="Mình chưa có phản hồi phù hợp.")
            history.add_messages([fallback])
            _trim_memory(history)
            return {"response": fallback}

        # ---------------------------------------------------------------------------
        # Graph wiring
        # ---------------------------------------------------------------------------

        graph = StateGraph(State)

        graph.add_node("planner", planner_node)
        graph.add_node("chat", chat_node)
        graph.add_node("sql_agent", sql_agent_node)
        graph.add_node("clarify", clarify_node)
        graph.add_node("inject_runtime_context", inject_runtime_context_node)
        graph.add_node("tools", tools_node)
        graph.add_node("finalize", finalize_node)

        # START → planner → one of four routes
        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "chat":      "chat",
                "sql_agent": "sql_agent",
                "finalize":  "finalize",
                "clarify":   "clarify",
            },
        )

        # chat → MCP tools or finalize
        graph.add_conditional_edges(
            "chat",
            tools_condition,
            {"tools": "inject_runtime_context", "__end__": "finalize"},
        )

        graph.add_edge("inject_runtime_context", "tools")
        graph.add_edge("tools", "chat")

        # sql_agent and clarify both land in finalize (memory + response)
        graph.add_edge("sql_agent", "finalize")
        graph.add_edge("clarify", "finalize")

        graph.add_edge("finalize", END)

        _main_graph = graph.compile()

    return _main_graph
