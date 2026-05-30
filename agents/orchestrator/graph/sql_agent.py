import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.orchestrator.llm import get_sql_llm
from .db import _build_database_dsn, TRANSACTIONS_DB_NAME, WALLETS_DB_NAME

load_dotenv()

POSTGRES_URI = os.getenv("POSTGRES_URI", "").strip()
WALLETS_DB = _build_database_dsn(WALLETS_DB_NAME) if POSTGRES_URI else ""
TRASACTIONS_DB = _build_database_dsn(TRANSACTIONS_DB_NAME) if POSTGRES_URI else ""

def get_current_day() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

SQL_AGENT_SYSTEM_PROMPT = f"""You are a SQL assistant for a PostgreSQL database.

Today is {get_current_day()}

Treat the schema below as the authoritative source for table definitions and types. Use this provided schema directly when composing SELECT queries or answering schema-related questions. Only consult or introspect the live database schema if a SQL execution fails when run against the real database.

If you cannot produce a correct answer or SQL based solely on the provided schema, respond with the exact token NEED_TOOL so the agent can retry using DB-introspection tools.

TABLE transactions (
    id UUID NOT NULL,
    amount NUMERIC(19, 2) NOT NULL,
    type VARCHAR(50) NOT NULL,
    category_id UUID,
    category_name VARCHAR(255),
    description VARCHAR(255),
    wallet_id UUID,
    user_id UUID NOT NULL,
    transaction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    image_url VARCHAR(1024),
    external_transaction_id VARCHAR(255),
    PRIMARY KEY (id)
)
TABLE wallets (
        id UUID NOT NULL, 
        name VARCHAR(255) NOT NULL, 
        balance NUMERIC(19, 2) NOT NULL, 
        currency VARCHAR(3) NOT NULL, 
        wallet_type VARCHAR(50) NOT NULL, 
        provider VARCHAR(255), 
        provider_account_id VARCHAR(255), 
        user_id UUID NOT NULL, 
        active BOOLEAN DEFAULT true NOT NULL, 
        created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
        CONSTRAINT wallets_pkey PRIMARY KEY (id)
)

TABLE categories (
    id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    icon VARCHAR(255),
    user_id UUID NOT NULL,
    budget_limit NUMERIC(19, 2),
    period VARCHAR(50),
    active BOOLEAN DEFAULT true NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    color VARCHAR(32),
    PRIMARY KEY (id)
)

Rules:
- Allow SELECT-only queries.
- Never write INSERT, UPDATE, DELETE, DROP, or other destructive statements.
- Use at most 5 rows unless the user explicitly asks for a larger sample.
- When the user asks for totals, counts, or summaries, use SQL tools instead of guessing.
- Bare finance queries like "lần chi tiêu gần nhất", "tổng chi tuần vừa rồi", or "số dư hiện tại" refer to the authenticated current user by default.
- If authenticated runtime context includes a current_user_id, use it and do not ask the user to provide their user_id again.
- Return a short answer in Vietnamese when the result is ready.
- If possible, return as Vietnamese table. Otherwise, return a concise Vietnamese summary.
"""


def _build_sql_context() -> tuple[SQLDatabase, Any, list[Any]]:
    if not POSTGRES_URI:
        raise RuntimeError("POSTGRES_URI is not set")
    
    if not WALLETS_DB or not TRASACTIONS_DB:
        raise RuntimeError("WALLETS_DB and TRASACTIONS_DB must be configured")

    model = get_sql_llm()
    
    # Create connections to both databases
    wallets_db = SQLDatabase.from_uri(WALLETS_DB)
    transactions_db = SQLDatabase.from_uri(TRASACTIONS_DB)
    
    # Create toolkits for each database
    wallets_toolkit = SQLDatabaseToolkit(db=wallets_db, llm=model)
    transactions_toolkit = SQLDatabaseToolkit(db=transactions_db, llm=model)
    
    # Combine all tools from both databases
    all_tools = wallets_toolkit.get_tools() + transactions_toolkit.get_tools()
    
    # Return wallets_db for dialect info, model, and combined tools
    return wallets_db, model, all_tools


def _build_sql_state_graph() -> StateGraph:
    db, model, tools = _build_sql_context()
    tool_node = ToolNode(tools)
    llm_with_tools = model.bind_tools(tools)

    async def sql_chat_node(state: MessagesState):
        messages = list(state.get("messages", []))

        # First, ask the base model (without tools) to answer using ONLY the provided schema.
        strict_system = SystemMessage(
            content=(
                f"{SQL_AGENT_SYSTEM_PROMPT}\nDialect: {db.dialect}\n"
                "StrictMode: Use only the provided schema. If you cannot answer or generate correct SQL based on this schema, respond with the exact token NEED_TOOL."
            )
        )

        try:
            response = await model.ainvoke([strict_system] + messages)
        except Exception:
            # If the model invocation fails for any reason, fall back to tool-enabled model.
            response = None

        # If the model explicitly asks for tools (by returning NEED_TOOL),
        # or the call failed, or the model returned a raw SQL statement,
        # use tools so the query is executed against the DB.
        need_tools = False
        if response is None:
            need_tools = True
        else:
            content = getattr(response, "content", "") or ""
            # If model asks for NEED_TOOL, or if it returned a SQL code block
            # or an explicit SELECT ... FROM statement, we should run with tools.
            if "NEED_TOOL" in content:
                need_tools = True
            else:
                import re as _re
                if "```sql" in content.lower() or _re.search(r"\bselect\b.+\bfrom\b", content, _re.IGNORECASE | _re.DOTALL):
                    need_tools = True

        if need_tools:
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
