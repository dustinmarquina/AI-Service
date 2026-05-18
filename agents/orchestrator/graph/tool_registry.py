"""
tool_registry.py

Holds a reference to the live MCP tools so the executor can call them
directly by name, bypassing the LLM tool-calling loop.

Populated once by build_main_graph() after the MCP client connects.
"""
from typing import Any

_tools: dict[str, Any] = {}


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
    return await tool.ainvoke(args)