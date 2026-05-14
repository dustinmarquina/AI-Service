from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def extract_faulty_sql_from_messages(messages: list[BaseMessage]) -> str | None:
    sql_calls: dict[str, str] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                if tool_call.get("name") != "sql_db_query":
                    continue

                query = tool_call.get("args", {}).get("query")
                tool_call_id = tool_call.get("id")
                if tool_call_id and isinstance(query, str) and query.strip():
                    sql_calls[tool_call_id] = query.strip()
            continue

        if not isinstance(message, ToolMessage):
            continue

        if getattr(message, "status", "success") != "error":
            continue

        faulty_sql = sql_calls.get(message.tool_call_id)
        if faulty_sql:
            return faulty_sql

    return None
