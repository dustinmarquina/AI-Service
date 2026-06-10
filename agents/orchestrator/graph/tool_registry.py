"""
tool_registry.py

Holds a reference to the live MCP tools so the executor can call them
directly by name, bypassing the LLM tool-calling loop.

Populated once by build_main_graph() after the MCP client connects.
"""
import json
from types import UnionType
from typing import Any

_tools: dict[str, Any] = {}


def _escape_prompt_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}").strip()


def _field_summary(tool: Any) -> str:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return ""

    fields = getattr(schema, "model_fields", None)
    if not isinstance(fields, dict):
        fields = getattr(schema, "__fields__", None)
    if not isinstance(fields, dict) or not fields:
        return ""

    parts: list[str] = []
    for field_name, field in fields.items():
        annotation = getattr(field, "annotation", None) or getattr(field, "outer_type_", None)
        required = getattr(field, "is_required", None)
        if callable(required):
            is_required = required()
        else:
            is_required = bool(getattr(field, "required", False))

        type_name = getattr(annotation, "__name__", None) or str(annotation or "Any")
        requirement = "required" if is_required else "optional"
        parts.append(f"{field_name}: {type_name} [{requirement}]")

    return ", ".join(parts)


def _tool_description(tool: Any) -> str:
    name = getattr(tool, "name", "unknown")
    description = str(getattr(tool, "description", "") or "").strip()
    fields = _field_summary(tool)
    schema = getattr(tool, "args_schema", None)
    has_optional_fields = False
    if schema is not None:
        model_fields = getattr(schema, "model_fields", None)
        if not isinstance(model_fields, dict):
            model_fields = getattr(schema, "__fields__", None)
        if isinstance(model_fields, dict):
            for field in model_fields.values():
                required = getattr(field, "is_required", None)
                if callable(required):
                    is_required = required()
                else:
                    is_required = bool(getattr(field, "required", False))
                if not is_required:
                    has_optional_fields = True
                    break

    signature = f"TOOL {name}({fields})" if fields else f"TOOL {name}()"
    lines = [signature]
    if description:
        lines.append(f"  {description}")
    if has_optional_fields:
        lines.append("  Optional fields may be omitted at planning time if they are not yet known.")
    return _escape_prompt_braces("\n".join(lines))


def _normalize_mcp_result(result: Any) -> Any:
    if isinstance(result, dict):
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
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


def _is_str_annotation(annotation: Any) -> bool:
    if annotation is str:
        return True

    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        return any(_is_str_annotation(arg) for arg in args if arg is not type(None))

    if isinstance(annotation, UnionType):
        return any(_is_str_annotation(arg) for arg in annotation.__args__ if arg is not type(None))

    return False


def _coerce_tool_args(tool: Any, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        return args

    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        fields = getattr(schema, "model_fields", None)
        if not isinstance(fields, dict):
            fields = getattr(schema, "__fields__", None)
        if isinstance(fields, dict):
            coerced = dict(args)
            for field_name, field in fields.items():
                if field_name not in coerced:
                    continue
                value = coerced[field_name]
                if value is None or isinstance(value, str):
                    continue

                annotation = getattr(field, "annotation", None) or getattr(field, "outer_type_", None)
                if _is_str_annotation(annotation):
                    coerced[field_name] = str(value)

            return coerced

    parameters = getattr(tool, "parameters", None)
    if not isinstance(parameters, dict):
        return args

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return args

    coerced = dict(args)
    for field_name, field_schema in properties.items():
        if field_name not in coerced or not isinstance(field_schema, dict):
            continue
        value = coerced[field_name]
        if value is None or isinstance(value, str):
            continue
        if field_schema.get("type") == "string":
            coerced[field_name] = str(value)

    return coerced


def register_tools(tools: list[Any]) -> None:
    """Called once at startup with the tools returned by MultiServerMCPClient."""
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            _tools[name] = tool


def describe_tools() -> str:
    """Return a prompt-safe description of the registered MCP tools."""
    if not _tools:
        return "- No MCP tools available"

    return "\n\n".join(_tool_description(tool) for tool in _tools.values())


def list_registered_tools() -> list[Any]:
    """Return the currently registered tool objects."""
    return list(_tools.values())


async def call_mcp_tool(name: str, args: dict) -> Any:
    """Invoke an MCP tool by name with resolved args. Raises KeyError if not found."""
    tool = _tools.get(name)
    if tool is None:
        raise KeyError(f"tool_registry: tool '{name}' not registered. Available: {list(_tools.keys())}")
    result = await tool.ainvoke(_coerce_tool_args(tool, args))
    return _normalize_mcp_result(result)
