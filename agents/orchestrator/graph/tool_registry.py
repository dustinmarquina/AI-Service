"""
tool_registry.py

Holds a reference to the live MCP tools so the executor can call them
directly by name, bypassing the LLM tool-calling loop.

Populated once by build_main_graph() after the MCP client connects.
"""
import json
from typing import Any

_tools: dict[str, Any] = {}


def _normalize_mcp_result(result: Any) -> Any:
    if isinstance(result, dict):
        return result

    if not isinstance(result, list) or len(result) != 1:
        return result

    item = result[0]
    if isinstance(item, dict):
        item_type = item.get("type")
        text = item.get("text")
    else:
        item_type = getattr(item, "type", None)
        text = getattr(item, "text", None)

    if item_type != "text" or not isinstance(text, str):
        return result

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return result

    return parsed if isinstance(parsed, dict) else result


def register_tools(tools: list[Any]) -> None:
    """Called once at startup with the tools returned by MultiServerMCPClient."""
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            _tools[name] = tool


async def call_mcp_tool(name: str, args: dict) -> Any:
    """Invoke an MCP tool by name with resolved args. Raises KeyError if not found."""
    tool = _tools.get(name)
    if tool is None:
        raise KeyError(f"tool_registry: tool '{name}' not registered. Available: {list(_tools.keys())}")
    result = await tool.ainvoke(args)
    return _normalize_mcp_result(result)
