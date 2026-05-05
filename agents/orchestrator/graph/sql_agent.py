import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.orchestrator.llm import get_sql_llm

load_dotenv()

SQL_AGENT_SYSTEM_PROMPT = """You are a SQL assistant for a PostgreSQL database.

Use the available SQL tools to inspect the schema and answer the user's question.
Rules:
- Prefer SELECT-only queries.
- Never write INSERT, UPDATE, DELETE, DROP, or other destructive statements.
- Use at most 5 rows unless the user explicitly asks for a larger sample.
- When the user asks for totals, counts, or summaries, use SQL tools instead of guessing.
- Return a short answer in Vietnamese when the result is ready.
"""


def _build_sql_context() -> tuple[SQLDatabase, Any, list[Any]]:
    postgres_uri = os.getenv("POSTGRES_URI", "").strip()
    if not postgres_uri:
        raise RuntimeError("POSTGRES_URI is not set")

    model = get_sql_llm()
    db = SQLDatabase.from_uri(postgres_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    return db, model, toolkit.get_tools()


def _build_sql_state_graph() -> StateGraph:
    db, model, tools = _build_sql_context()
    tool_node = ToolNode(tools)
    llm_with_tools = model.bind_tools(tools)

    async def sql_chat_node(state: MessagesState):
        messages = list(state.get("messages", []))
        response = await llm_with_tools.ainvoke(
            [SystemMessage(content=f"{SQL_AGENT_SYSTEM_PROMPT}\nDialect: {db.dialect}")] + messages
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", sql_chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges(
        "chat",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "chat")

    return graph


def build_sql_agent_graph():
    return _build_sql_state_graph().compile()


# ---------------------------------------------------------------------------
# Subgraph entry-point used by the main graph
# Accepts MessagesState, returns {"messages": [...]} with the final AI answer
# ---------------------------------------------------------------------------
_sql_agent: Any = None


def get_sql_agent():
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = build_sql_agent_graph()
    return _sql_agent