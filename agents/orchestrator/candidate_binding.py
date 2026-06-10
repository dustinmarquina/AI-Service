import json
import re
import unicodedata
from typing import Any

_LOWEST_HINTS = {
    "lowest", "smallest", "minimum", "least", "min",
    "thap nhat", "nho nhat", "it nhat", "thieu nhat",
}
_HIGHEST_HINTS = {
    "highest", "largest", "maximum", "most", "max",
    "cao nhat", "lon nhat", "nhieu nhat",
}
_FIELD_ALIASES: dict[str, set[str]] = {
    "balance": {"balance", "so du", "funds"},
    "amount": {"amount", "so tien", "gia", "price", "cost", "chi tieu", "expense", "spend"},
    "score": {"score", "diem"},
    "total": {"total", "tong"},
    "count": {"count", "so luong"},
    "quantity": {"quantity", "so luong"},
}


class _FallbackSystemMessage:
    def __init__(self, content: str):
        self.content = content


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    if not parts:
        return value
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def candidate_value(row: dict[str, Any], arg_name: str) -> object:
    base_name = arg_name[:-3] if arg_name.endswith("_id") else arg_name
    for key in (arg_name, _snake_to_camel(arg_name), "id", f"{base_name}Id"):
        value = row.get(key)
        if value:
            return value
    return None


def row_contains_value(row: dict[str, Any], value: Any, arg_name: str) -> bool:
    resolved = candidate_value(row, arg_name)
    if resolved is None:
        return False
    return str(resolved) == str(value)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(value: str) -> set[str]:
    normalized = _normalize_text(value)
    return {token for token in normalized.split() if token}


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _comparative_direction(user_text: str) -> str | None:
    normalized = _normalize_text(user_text)
    tokens = _tokenize(user_text)
    if any(keyword in normalized for keyword in _LOWEST_HINTS) or tokens & _LOWEST_HINTS:
        return "min"
    if any(keyword in normalized for keyword in _HIGHEST_HINTS) or tokens & _HIGHEST_HINTS:
        return "max"
    return None


def _common_numeric_fields(rows: list[dict[str, Any]]) -> set[str]:
    common: set[str] | None = None
    for row in rows:
        numeric_fields = {
            key for key, value in row.items()
            if not key.endswith("_id")
            and not key.lower().endswith("id")
            and _numeric_value(value) is not None
        }
        common = numeric_fields if common is None else common & numeric_fields
    return common or set()


def _field_from_user_text(rows: list[dict[str, Any]], user_text: str) -> str | None:
    common_fields = _common_numeric_fields(rows)
    if not common_fields:
        return None

    normalized = _normalize_text(user_text)
    for field in common_fields:
        aliases = _FIELD_ALIASES.get(field, {field.replace("_", " ")})
        if any(alias in normalized for alias in aliases):
            return field

    if len(common_fields) == 1:
        return next(iter(common_fields))
    return None


def _select_by_comparative_hint(rows: list[dict[str, Any]], user_text: str) -> dict[str, Any] | None:
    direction = _comparative_direction(user_text)
    if direction is None:
        return None

    field = _field_from_user_text(rows, user_text)
    if field is None:
        return None

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        numeric = _numeric_value(row.get(field))
        if numeric is None:
            continue
        scored.append((numeric, row))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=(direction == "max"))
    best_value = scored[0][0]
    top_rows = [row for value, row in scored if value == best_value]
    if len(top_rows) == 1:
        return top_rows[0]
    return None


def build_candidate_binding_prompt(
    *,
    user_text: str,
    current_context: dict[str, Any],
    arg_name: str,
    rows: list[dict[str, Any]],
) -> str:
    return (
        "You are selecting a value for a missing tool argument from candidate SQL rows.\n"
        "Given the original user request, the current tool context, the missing argument name, and the candidate rows,\n"
        "decide whether one candidate row should be bound to that argument or whether clarification is required.\n"
        "Output ONLY valid JSON.\n\n"
        "Allowed outputs:\n"
        '- {"route":"execute","reason":"...","bindings":{"ARG_NAME":"candidate-id"}}\n'
        '- {"route":"clarify","reason":"...","message":"..."}\n\n'
        "Rules:\n"
        f"- You may bind only the missing argument '{arg_name}'.\n"
        "- The binding value must come from one of the provided candidate rows.\n"
        "- Use the candidate row data when the user implies a comparative choice such as lowest, highest, most recent, or largest.\n"
        "- If multiple candidate rows remain plausible, return clarify instead of guessing.\n"
        "- Do not invent IDs or values that do not appear in the candidate rows.\n\n"
        f"User request: {json.dumps(user_text, ensure_ascii=False)}\n"
        f"Current context: {json.dumps(current_context, ensure_ascii=False)}\n"
        f"Missing argument: {json.dumps(arg_name, ensure_ascii=False)}\n"
        f"Candidate rows: {json.dumps(rows, ensure_ascii=False)}"
    )


def parse_candidate_binding_output(text: str) -> dict[str, Any]:
    clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
    data = json.loads(clean)
    route = data.get("route")
    if route not in ("execute", "clarify"):
        raise ValueError(f"unsupported candidate binding route: {route}")
    return data


async def select_candidate_binding(
    llm: Any,
    *,
    user_text: str,
    current_context: dict[str, Any],
    arg_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    deterministic_row = _select_by_comparative_hint(rows, user_text)
    if deterministic_row is not None:
        value = candidate_value(deterministic_row, arg_name)
        if value is not None:
            return {"route": "execute", "value": value}

    try:
        try:
            from langchain_core.messages import SystemMessage
        except Exception:
            SystemMessage = _FallbackSystemMessage
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=build_candidate_binding_prompt(
                        user_text=user_text,
                        current_context=current_context,
                        arg_name=arg_name,
                        rows=rows,
                    )
                )
            ]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_candidate_binding_output(raw)
    except Exception:
        return None

    if parsed["route"] == "clarify":
        return {
            "route": "clarify",
            "message": str(parsed.get("message") or "I need more information to choose the correct option."),
        }

    bindings = parsed.get("bindings") or {}
    if not isinstance(bindings, dict):
        return None

    bound_value = bindings.get(arg_name)
    if bound_value is None:
        return None

    if not any(row_contains_value(row, bound_value, arg_name) for row in rows):
        return None

    return {"route": "execute", "value": bound_value}
