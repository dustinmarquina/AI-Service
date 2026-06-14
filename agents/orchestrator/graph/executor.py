import json
import logging
import os
import re
import unicodedata
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from .state import State

logger = logging.getLogger(__name__)
_selection_llm = None


class _FallbackSystemMessage:
    def __init__(self, content: str):
        self.content = content

# ---------------------------------------------------------------------------
# Placeholder resolution
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"^\$(\w+)\.(\w+)$")
_EMBEDDED_PLACEHOLDER = re.compile(r"\$(\w+)\.(\w+)")


def _resolve_value(value: Any, step_results: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    m = _PLACEHOLDER.match(value.strip())
    if not m:
        return value
    step_id, field = m.group(1), m.group(2)
    result = step_results.get(step_id)
    if result is None:
        raise ValueError(f"executor: step '{step_id}' has no result yet")
    if isinstance(result, dict):
        if field not in result:
            raise ValueError(
                f"executor: field '{field}' not found in step '{step_id}': {list(result.keys())}"
            )
        return result[field]
    raise ValueError(f"executor: step '{step_id}' result is not a dict: {type(result)}")


def _resolve_args(args: dict, step_results: dict[str, Any]) -> dict:
    return {k: _resolve_value(v, step_results) for k, v in args.items()}


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _resolve_query_hint(query_hint: str, step_results: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        step_id, field = match.group(1), match.group(2)
        result = step_results.get(step_id)
        if not isinstance(result, dict):
            raise ValueError(f"executor: step '{step_id}' has no dict result for SQL placeholder resolution")
        if field not in result:
            raise ValueError(
                f"executor: field '{field}' not found in step '{step_id}' for SQL placeholder resolution"
            )
        return _sql_literal(result[field])

    return _EMBEDDED_PLACEHOLDER.sub(replace, query_hint)


_RESERVED_RUNTIME_KEYS = {"token", "user_id"}


def _append_past_step(
    past_steps: list[dict[str, Any]],
    *,
    step_id: str,
    step_type: str,
    step_input: dict[str, Any],
    output: Any,
    status: str,
    summary: str,
    reasoning: str | None = None,
) -> list[dict[str, Any]]:
    step_record = {
        "step_id": step_id,
        "type": step_type,
        "input": step_input,
        "output": output,
        "status": status,
        "summary": summary,
    }
    if reasoning:
        step_record["reasoning"] = reasoning
    return past_steps + [step_record]


def _runtime_user_id(state: State, step_results: dict[str, Any]) -> str:
    state_user_id = str(state.get("user_id") or "").strip()
    if state_user_id:
        return state_user_id

    bootstrap_result = step_results.get("s0")
    if isinstance(bootstrap_result, dict):
        candidate = str(bootstrap_result.get("user_id") or "").strip()
        if candidate:
            return candidate

    for result in step_results.values():
        if not isinstance(result, dict):
            continue
        candidate = str(result.get("user_id") or "").strip()
        if candidate:
            return candidate

    return str(os.getenv("TRANSACTION_USER_ID", "")).strip()


def _inject_runtime_args(
    tool_name: str,
    args: dict[str, Any],
    state: State,
    step_results: dict[str, Any],
) -> dict[str, Any]:
    resolved_args = dict(args)
    runtime_token = str(state.get("token") or os.getenv("TRANSACTION_API_TOKEN", "")).strip()
    runtime_user_id = _runtime_user_id(state, step_results)
    latest_user_text = ""
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, AIMessage):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            latest_user_text = content.strip()
            break
    if runtime_token and "token" not in resolved_args:
        resolved_args["token"] = runtime_token
    if runtime_user_id and tool_name != "get_user_id" and "user_id" not in resolved_args:
        resolved_args["user_id"] = runtime_user_id
    if tool_name == "create_transaction" and latest_user_text and "request_text" not in resolved_args:
        resolved_args["request_text"] = latest_user_text
    return resolved_args


def _status_from_tool_result(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("status") or "success")
    return "success"


def _summary_from_tool_result(tool_name: str, result: Any) -> str:
    if isinstance(result, dict):
        if result.get("status") == "error":
            return str(result.get("message") or f"{tool_name} failed")
        if result.get("message"):
            return str(result["message"])
    return f"{tool_name} completed"


# ---------------------------------------------------------------------------
# Generic row selection
# ---------------------------------------------------------------------------

_ORDINAL_INDEX_MAP = {
    "first": 1,
    "1st": 1,
    "dau tien": 1,
    "thu nhat": 1,
    "second": 2,
    "2nd": 2,
    "thu hai": 2,
    "third": 3,
    "3rd": 3,
    "thu ba": 3,
    "fourth": 4,
    "4th": 4,
    "thu tu": 4,
    "fifth": 5,
    "5th": 5,
    "thu nam": 5,
}


def _get_selection_llm():
    global _selection_llm
    if _selection_llm is None:
        from agents.orchestrator.llm import get_classifier_llm
        _selection_llm = get_classifier_llm()
    return _selection_llm

def _row_label(row: dict[str, Any]) -> str:
    return str(
        row.get("label")
        or row.get("name")
        or row.get("title")
        or row.get("id")
        or "option"
    )


def _row_detail(row: dict[str, Any]) -> str:
    extras: list[str] = []
    for key in ("balance", "currency", "status", "type"):
        value = row.get(key)
        if value not in (None, "", []):
            extras.append(str(value))
    return " | ".join(extras)


def _build_prompt(rows: list[dict], prompt: str | None = None, input_kind: str = "selection", field: str | None = None) -> str:
    if input_kind == "text":
        field_name = field or "input"
        return prompt or f"Please provide the {field_name}:"
    lines = [prompt or "Please choose one option:"]
    for i, row in enumerate(rows, 1):
        label = _row_label(row)
        detail = _row_detail(row)
        suffix = f" — {detail}" if detail else ""
        lines.append(f"  {i}. {label}{suffix}")
    lines.append("\nReply with the number or the option name:")
    return "\n".join(lines)


def _selection_prompt_intro(pending_input: dict[str, Any], default: str) -> str:
    stored_intro = str(pending_input.get("prompt_intro") or "").strip()
    if stored_intro:
        return stored_intro
    stored_prompt = str(pending_input.get("prompt") or "").strip()
    if stored_prompt:
        marker = "\nReply with the number or the option name:"
        if marker in stored_prompt:
            return stored_prompt.split(marker, 1)[0].strip()
        return stored_prompt
    return default


def _extract_selection_text(selection: Any) -> str:
    if isinstance(selection, dict):
        for value in selection.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(selection, str):
        return selection.strip()
    return ""


def _normalize_choice_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_selection_index(text: str, row_count: int) -> int | None:
    normalized = _normalize_choice_text(text)
    if not normalized:
        return None

    if normalized.isdigit():
        index = int(normalized)
        if 1 <= index <= row_count:
            return index - 1
        return None

    for pattern in (r"\b(?:option|lua chon|chon|thu)\s+(\d+)\b", r"\bcai thu\s+(\d+)\b"):
        match = re.search(pattern, normalized)
        if match:
            index = int(match.group(1))
            if 1 <= index <= row_count:
                return index - 1

    for phrase, index in _ORDINAL_INDEX_MAP.items():
        if phrase in normalized and 1 <= index <= row_count:
            return index - 1

    return None


def _row_search_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("label", "name", "title", "id"):
        value = row.get(key)
        if value not in (None, ""):
            texts.append(_normalize_choice_text(str(value)))
    return [text for text in texts if text]


def _score_row_match(normalized_selection: str, row: dict[str, Any]) -> int:
    if not normalized_selection:
        return 0

    selection_tokens = set(normalized_selection.split())
    best_score = 0
    for candidate_text in _row_search_texts(row):
        candidate_tokens = set(candidate_text.split())
        if normalized_selection == candidate_text:
            best_score = max(best_score, 100)
            continue
        if selection_tokens and selection_tokens == candidate_tokens:
            best_score = max(best_score, 95)
            continue
        if candidate_text.startswith(normalized_selection) or normalized_selection.startswith(candidate_text):
            best_score = max(best_score, 85)
            continue

        overlap = len(selection_tokens & candidate_tokens)
        if overlap and selection_tokens:
            ratio = overlap / len(selection_tokens)
            if overlap == len(selection_tokens):
                best_score = max(best_score, 75)
            elif ratio >= 0.7:
                best_score = max(best_score, 65)
            elif ratio >= 0.5:
                best_score = max(best_score, 55)

        if normalized_selection in candidate_text or candidate_text in normalized_selection:
            best_score = max(best_score, 50)

    return best_score


def _resolve_row_selection(selection: Any, rows: list[dict]) -> dict | None:
    selection_text = _extract_selection_text(selection)
    if not selection_text:
        return None 

    index = _parse_selection_index(selection_text, len(rows))
    if index is not None:
        return rows[index]

    normalized_selection = _normalize_choice_text(selection_text)
    if not normalized_selection:
        return None

    scored: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        score = _score_row_match(normalized_selection, row)
        if score > 0:
            scored.append((score, row))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_row = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1
    if best_score < 60:
        return None
    if second_score >= best_score - 10:
        return None

    return best_row


def _build_selection_llm_prompt(selection_text: str, rows: list[dict], prompt_intro: str) -> str:
    candidates = [
        {"index": index + 1, "label": _row_label(row), "data": row}
        for index, row in enumerate(rows)
    ]
    return (
        "You are a strict selection resolver.\n"
        "Choose exactly one candidate only when the user's reply clearly maps to it.\n"
        "If the reply is ambiguous or weak, return clarify.\n"
        "Output JSON only.\n\n"
        "Allowed outputs:\n"
        '- {"route":"select","index":2,"reason":"..."}\n'
        '- {"route":"clarify","reason":"..."}\n\n'
        f"Prompt: {json.dumps(prompt_intro, ensure_ascii=False)}\n"
        f"Selection reply: {json.dumps(selection_text, ensure_ascii=False)}\n"
        f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
    )


def _parse_selection_llm_output(text: str) -> dict[str, Any]:
    clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
    data = json.loads(clean)
    route = data.get("route")
    if route not in ("select", "clarify"):
        raise ValueError(f"unsupported selection route: {route}")
    return data


async def _resolve_row_selection_with_llm(selection: Any, rows: list[dict], prompt_intro: str) -> dict | None:
    selection_text = _extract_selection_text(selection)
    if not selection_text:
        return None

    llm = _get_selection_llm()
    try:
        try:
            from langchain_core.messages import SystemMessage
        except Exception:
            SystemMessage = _FallbackSystemMessage
        response = await llm.ainvoke(
            [SystemMessage(content=_build_selection_llm_prompt(selection_text, rows, prompt_intro))]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_selection_llm_output(raw)
    except Exception:
        return None

    if parsed["route"] != "select":
        return None

    index = parsed.get("index")
    if not isinstance(index, int):
        return None
    if not (1 <= index <= len(rows)):
        return None

    return rows[index - 1]


def _selection_value(row: dict[str, Any], value_field: str | None, selection_field: str | None) -> Any:
    if value_field and row.get(value_field) not in (None, ""):
        return row.get(value_field)
    if selection_field and row.get(selection_field) not in (None, ""):
        return row.get(selection_field)
    if row.get("id") not in (None, ""):
        return row.get("id")
    return None


def _format_result_row(row: dict[str, Any]) -> str:
    parts: list[str] = []
    preferred_keys = ("description", "amount", "transaction_date", "name", "balance", "currency", "id")
    seen: set[str] = set()

    for key in preferred_keys:
        value = row.get(key)
        if value not in (None, "", []):
            parts.append(f"{key}={value}")
            seen.add(key)

    for key, value in row.items():
        if key in seen or value in (None, "", []):
            continue
        parts.append(f"{key}={value}")

    return " | ".join(parts) or str(row)


def _format_result_rows(rows: list[dict[str, Any]], limit: int = 10) -> list[str]:
    lines: list[str] = []
    for idx, row in enumerate(rows[:limit], 1):
        lines.append(f"{idx}. {_format_result_row(row)}")
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more")
    return lines


async def _prompt_for_selection(prompt_intro: str, candidates: list[dict]) -> dict | None:
    prompt = _build_prompt(candidates, prompt_intro)
    while True:
        selection = interrupt(prompt)
        if selection == prompt:
            return None

        matched = _resolve_row_selection(selection, candidates)
        if matched is None:
            matched = await _resolve_row_selection_with_llm(selection, candidates, prompt_intro)
        if matched is not None:
            return matched

        prompt = _build_prompt(
            candidates,
            f"{prompt_intro}\nInvalid selection. Please choose one of the listed options.",
        )
async def _prompt_for_typing(prompt: str) -> str | None:
    while True:
        user_input = interrupt(prompt)
        if user_input == prompt:
            return None
        if isinstance(user_input, str) and user_input.strip():
            return user_input.strip()
        prompt = f"{prompt}\nInput cannot be empty. Please provide a valid input."

# ---------------------------------------------------------------------------
# Executor node
# ---------------------------------------------------------------------------

async def executor_node(state: State) -> dict:
    steps: list[dict] = state.get("steps") or []
    step_index: int = state.get("step_index") or 0
    step_results: dict = dict(state.get("step_results") or {})
    past_steps: list[dict[str, Any]] = list(state.get("past_steps") or [])

    if step_index >= len(steps):
        logger.warning("executor: step_index=%d >= len(steps)=%d", step_index, len(steps))
        return {"step_index": step_index, "past_steps": past_steps}

    current = steps[step_index]
    step_id = current["id"]
    step_type = current["type"]
    step_reasoning = str(current.get("reasoning") or "").strip()

    logger.info(
        "executor: step %d/%d id=%s type=%s reasoning=%s",
        step_index + 1, len(steps), step_id, step_type, step_reasoning or "<none>",
    )

    # ── SQL step — direct asyncpg, no LLM ────────────────────────────────────
    if step_type == "sql":
        query_hint = current.get("query_hint", "")
        description = current.get("description", "")
        selection_mode = str(current.get("selection_mode") or "").strip().lower() or "none"

        user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()
        if user_id:
            query_hint = query_hint.replace(":user_id", f"\'{user_id}\'")
        if "$" in query_hint:
            query_hint = _resolve_query_hint(query_hint, step_results)

        logger.info("executor: sql step %s context description=%s user_id=%s", step_id, description, user_id)
        logger.info("executor: sql step %s query_hint(raw)=%s", step_id, current.get("query_hint", ""))
        logger.info("executor: sql step %s query_hint(resolved)=%s", step_id, query_hint)

        if not query_hint:
            logger.error("executor: sql step %s has no query_hint, cannot execute", step_id)
            past_steps = _append_past_step(
                past_steps,
                step_id=step_id,
                step_type="sql",
                step_input={"query_hint": current.get("query_hint", "")},
                output=None,
                status="error",
                summary=f"SQL step {step_id} missing query_hint",
                reasoning=step_reasoning,
            )
            return {
                "messages": [AIMessage(content=f"Lỗi: bước {step_id} thiếu câu truy vấn.")],
                "step_index": len(steps),
                "step_results": step_results,
                "past_steps": past_steps,
            }

        from .db import execute_query
        try:
            rows = await execute_query(query_hint)
        except Exception as exc:
            logger.error("executor: sql step %s failed: %s", step_id, exc)
            past_steps = _append_past_step(
                past_steps,
                step_id=step_id,
                step_type="sql",
                step_input={"query_hint": query_hint},
                output={"error": str(exc)},
                status="error",
                summary=str(exc),
                reasoning=step_reasoning,
            )
            return {
                "messages": [AIMessage(content=f"Lỗi truy vấn dữ liệu: {exc}")],
                "step_index": len(steps),
                "step_results": step_results,
                "past_steps": past_steps,
            }

        if not rows:
            logger.warning("executor: sql step %s returned no rows", step_id)
            past_steps = _append_past_step(
                past_steps,
                step_id=step_id,
                step_type="sql",
                step_input={"query_hint": query_hint},
                output=[],
                status="empty",
                summary="No rows returned",
                reasoning=step_reasoning,
            )
            return {
                "messages": [AIMessage(content="Không tìm thấy dữ liệu phù hợp.")],
                "step_index": len(steps),
                "step_results": step_results,
                "past_steps": past_steps,
            }

        pending_input = step_results.get(f"{step_id}__pending_input")
        if pending_input and selection_mode == "required":
            candidates: list[dict] = pending_input.get("candidates", [])
            prompt_intro = _selection_prompt_intro(pending_input, "Please choose one option:")
            matched = await _prompt_for_selection(prompt_intro, candidates)
            if matched is None:
                return {
                    "step_index": step_index,
                    "step_results": step_results,
                    "past_steps": past_steps,
                }

            step_results.pop(f"{step_id}__pending_input", None)
            step_results[step_id] = matched
            past_steps = _append_past_step(
                past_steps,
                step_id=step_id,
                step_type="sql",
                step_input={"query_hint": query_hint},
                output=matched,
                status="success",
                summary=f"Selected one option from {len(candidates)} candidates",
                reasoning=step_reasoning,
            )
            return {
                "step_index": step_index + 1,
                "step_results": step_results,
                "past_steps": past_steps,
            }

        if selection_mode != "required":
            output = rows[0] if len(rows) == 1 else rows
            step_results[step_id] = output
            row_word = "row" if len(rows) == 1 else "rows"
            past_steps = _append_past_step(
                past_steps,
                step_id=step_id,
                step_type="sql",
                step_input={"query_hint": query_hint},
                output=output,
                status="success",
                summary=f"SQL step {step_id} returned {len(rows)} {row_word}",
                reasoning=step_reasoning,
            )
            return {
                "step_index": step_index + 1,
                "step_results": step_results,
                "past_steps": past_steps,
            }

        if len(rows) == 1:
            step_results[step_id] = rows[0]
            logger.info("executor: step %s single row, auto-selected: %s", step_id, rows[0])
            past_steps = _append_past_step(
                past_steps,
                step_id=step_id,
                step_type="sql",
                step_input={"query_hint": query_hint},
                output=rows[0],
                status="success",
                summary=f"SQL step {step_id} returned 1 row",
                reasoning=step_reasoning,
            )
            return {
                "step_index": step_index + 1,
                "step_results": step_results,
                "past_steps": past_steps,
            }

        prompt = _build_prompt(rows, current.get("selection_prompt"))
        step_results[f"{step_id}__pending_input"] = {
            "prompt_intro": current.get("selection_prompt") or "Please choose one option:",
            "prompt": prompt,
            "candidates": rows,
        }
        logger.info("executor: interrupting for generic selection, %d options", len(rows))
        matched = await _prompt_for_selection(
            step_results[f"{step_id}__pending_input"]["prompt_intro"],
            rows,
        )
        if matched is None:
            return {
                "step_index": step_index,
                "step_results": step_results,
                "past_steps": past_steps,
            }

        step_results.pop(f"{step_id}__pending_input", None)
        step_results[step_id] = matched
        past_steps = _append_past_step(
            past_steps,
            step_id=step_id,
            step_type="sql",
            step_input={"query_hint": query_hint},
            output=matched,
            status="success",
            summary=f"Selected one option from {len(rows)} candidates",
            reasoning=step_reasoning,
        )
        return {
            "step_index": step_index + 1,
            "step_results": step_results,
            "past_steps": past_steps,
        }

    # ── Tool step ─────────────────────────────────────────────────────────────
    elif step_type == "tool":
        tool_name = current.get("name", "")
        raw_args = current.get("args", {})
        write_to_state = current.get("write_to_state")  # e.g. "user_id"
        pending_input = step_results.get(f"{step_id}__pending_input")

        if pending_input:
            input_kind = str(pending_input.get("input_kind", "selection")).strip().lower()
            if input_kind == "selection":
                candidates = list(pending_input.get("candidates") or [])
                prompt_intro = _selection_prompt_intro(
                    pending_input,
                    "Please choose one option:",
                )
                selection_field = str(pending_input.get("selection_field") or "").strip()
                value_field = str(pending_input.get("value_field") or "").strip() or None
                matched = await _prompt_for_selection(prompt_intro, candidates)
                if matched is None: 
                    return {
                        "step_index": step_index,
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }

                selected_value = _selection_value(matched, value_field, selection_field)
                if selection_field and selected_value not in (None, ""):
                    resolved_args = dict(pending_input.get("args") or {})
                    resolved_args[selection_field] = selected_value
                else:
                    past_steps = _append_past_step(
                        past_steps,
                        step_id=step_id,
                        step_type="tool",
                        step_input=pending_input.get("args", {}),
                        output=matched,
                        status="error",
                        summary="Selected option did not provide a usable value",
                        reasoning=step_reasoning,
                    )
                    return {
                        "messages": [AIMessage(content="The selected option could not be used.")],
                        "step_index": len(steps),
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }
            if input_kind == "text":
                field = str(pending_input.get("field") or "").strip()
                if field:
                    resolved_args = dict(pending_input.get("args") or {})
                    resolved_args[field] = selected_value
                else:
                    past_steps = _append_past_step(
                        past_steps,
                        step_id=step_id,
                        step_type="tool",
                        step_input=pending_input.get("args", {}),
                        output=matched,
                        status="error",
                        summary="Tool requested text input without specifying the field",
                        reasoning=step_reasoning,
                    )
                    return {
                        "messages": [AIMessage(content="The tool requested input in an invalid format.")],
                        "step_index": len(steps),
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }
            step_results.pop(f"{step_id}__pending_input", None)
        else:
            try:
                resolved_args = {
                    k: _resolve_value(v, step_results)
                    for k, v in raw_args.items()
                    if k not in _RESERVED_RUNTIME_KEYS
                }
            except ValueError as exc:
                logger.error("executor: placeholder resolution failed: %s", exc)
                past_steps = _append_past_step(
                    past_steps,
                    step_id=step_id,
                    step_type="tool",
                    step_input=raw_args,
                    output={"error": str(exc)},
                    status="error",
                    summary=str(exc),
                    reasoning=step_reasoning,
                )
                return {
                    "messages": [AIMessage(content=f"Lỗi xử lý bước {step_id}: {exc}")],
                    "step_index": len(steps),
                    "step_results": step_results,
                    "past_steps": past_steps,
                }

            resolved_args = _inject_runtime_args(tool_name, resolved_args, state, step_results)

        logger.info("executor: calling tool '%s' args=%s", tool_name, {
            k: (v[:8] + "...") if k == "token" and isinstance(v, str) else v
            for k, v in resolved_args.items()
        })

        from .tool_registry import call_mcp_tool
        tool_result = await call_mcp_tool(tool_name, resolved_args)
        if isinstance(tool_result, dict) and tool_result.get("status") == "needs_input":
            input_kind = str(tool_result.get("input_kind", "selection")).strip().lower()
            if input_kind == "selection":
                candidates = list(tool_result.get("candidates") or [])
                selection_field = str(tool_result.get("selection_field") or "").strip()
                prompt_intro = str(tool_result.get("prompt") or "Please choose one option:")
                prompt = _build_prompt(candidates, prompt_intro)
                if not candidates or not selection_field:
                    past_steps = _append_past_step(
                        past_steps,
                        step_id=step_id,
                        step_type="tool",
                        step_input=resolved_args,
                        output=tool_result,
                        status="error",
                        summary="Tool requested input without a valid selection contract",
                        reasoning=step_reasoning,
                    )
                    return {
                        "messages": [AIMessage(content="The tool requested input in an invalid format.")],
                        "step_index": len(steps),
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }

                step_results[f"{step_id}__pending_input"] = {
                    "prompt_intro": prompt_intro,
                    "prompt": prompt,
                    "candidates": candidates,
                    "selection_field": selection_field,
                    "value_field": tool_result.get("value_field"),
                    "args": resolved_args,
                }
                matched = await _prompt_for_selection(prompt_intro, candidates)
                if matched is None:
                    return {
                        "step_index": step_index,
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }
                selected_value = _selection_value(
                    matched,
                    str(tool_result.get("value_field") or "").strip() or None,
                    selection_field,
                )
                if selected_value in (None, ""):
                    past_steps = _append_past_step(
                        past_steps,
                        step_id=step_id,
                        step_type="tool",
                        step_input=resolved_args,
                        output=matched,
                        status="error",
                        summary="Selected option did not provide a usable value",
                        reasoning=step_reasoning,
                    )
                    return {
                        "messages": [AIMessage(content="The selected option could not be used.")],
                        "step_index": len(steps),
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }
                resolved_args = dict(resolved_args)
                resolved_args[selection_field] = selected_value
            elif input_kind == "text":
                field = str(tool_result.get("field") or "").strip()
                prompt_intro = str(tool_result.get("prompt") or "Please provide the missing input:")
                if not field:
                    past_steps = _append_past_step(
                        past_steps,
                        step_id=step_id,
                        step_type="tool",
                        step_input=resolved_args,
                        output=tool_result,
                        status="error",
                        summary="Tool requested text input without specifying the field",
                        reasoning=step_reasoning,
                    )
                    return {
                        "messages": [AIMessage(content="The tool requested input in an invalid format.")],
                        "step_index": len(steps),
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }
                step_results[f"{step_id}__pending_input"] = {
                    "input_kind": "text",
                    "field": field,
                    "prompt_intro": prompt_intro,
                    "prompt": prompt_intro,
                    "args": resolved_args,
                }
                user_input = await _prompt_for_typing(prompt_intro)
                if user_input is None:
                    return {
                        "step_index": step_index,
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }
                selected_value = user_input.strip()
                resolved_args = dict(resolved_args)
                resolved_args[field] = selected_value
            
            
            step_results.pop(f"{step_id}__pending_input", None)
            logger.info("executor: calling tool '%s' args=%s", tool_name, {
                k: (v[:8] + "...") if k == "token" and isinstance(v, str) else v
                for k, v in resolved_args.items()
            })
            tool_result = await call_mcp_tool(tool_name, resolved_args)
            if isinstance(tool_result, dict) and tool_result.get("status") == "needs_input":
                input_kind = str(tool_result.get("input_kind", "selection")).strip().lower()
                if input_kind == "selection":
                    step_results[f"{step_id}__pending_input"] = {
                        "prompt_intro": str(tool_result.get("prompt") or prompt_intro),
                        "prompt": _build_prompt(
                            list(tool_result.get("candidates") or []),
                            str(tool_result.get("prompt") or prompt_intro),
                        ),
                        "candidates": list(tool_result.get("candidates") or []),
                        "selection_field": str(tool_result.get("selection_field") or selection_field).strip(),
                        "value_field": tool_result.get("value_field"),
                        "args": resolved_args,
                    }
                    return {
                        "step_index": step_index,
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }
                if input_kind == "text":
                    step_results[f"{step_id}__pending_input"] = {
                        "input_kind": "text",
                        "field": str(tool_result.get("field") or field).strip(),
                        "prompt_intro": str(tool_result.get("prompt") or prompt_intro),
                        "prompt": str(tool_result.get("prompt") or prompt_intro),
                        "args": resolved_args,
                    }
                    return {
                        "step_index": step_index,
                        "step_results": step_results,
                        "past_steps": past_steps,
                    }

            step_results[step_id] = tool_result
            past_steps = _append_past_step(
                past_steps,
                step_id=step_id,
                step_type="tool",
                step_input=resolved_args,
                output=tool_result,
                status=_status_from_tool_result(tool_result),
                summary=_summary_from_tool_result(tool_name, tool_result),
                reasoning=step_reasoning,
            )
            state_update: dict = {
                "step_index": step_index + 1,
                "step_results": step_results,
                "past_steps": past_steps,
            }
            if write_to_state and isinstance(tool_result, dict) and tool_result.get("status") == "success":
                value = tool_result.get(write_to_state)
                if value:
                    state_update[write_to_state] = value
            return state_update

        step_results[step_id] = tool_result

        logger.info("executor: step %s tool result=%s", step_id, str(tool_result)[:200])
        past_steps = _append_past_step(
            past_steps,
            step_id=step_id,
            step_type="tool",
            step_input=resolved_args,
            output=tool_result,
            status=_status_from_tool_result(tool_result),
            summary=_summary_from_tool_result(tool_name, tool_result),
            reasoning=step_reasoning,
        )

        # write_to_state: propagate a result field directly into graph state
        # e.g. get_user_id writes user_id so subsequent SQL steps can use :user_id
        state_update: dict = {
            "step_index": step_index + 1,
            "step_results": step_results,
            "past_steps": past_steps,
        }
        if write_to_state and tool_result.get("status") == "success":
            value = tool_result.get(write_to_state)
            if value:
                logger.info("executor: write_to_state %s=%s", write_to_state, value)
                state_update[write_to_state] = value
            else:
                logger.warning(
                    "executor: write_to_state '%s' not found in result: %s",
                    write_to_state, tool_result,
                )

        return state_update

    else:
        logger.error("executor: unknown step type '%s'", step_type)
        past_steps = _append_past_step(
            past_steps,
            step_id=step_id,
            step_type=str(step_type),
            step_input=current,
            output=None,
            status="error",
            summary=f"Unknown step type: {step_type}",
            reasoning=step_reasoning,
        )
        return {
            "step_index": step_index + 1,
            "step_results": step_results,
            "past_steps": past_steps,
        }


def should_continue_executing(state: State) -> str:
    steps = state.get("steps") or []
    step_index = state.get("step_index") or 0
    if step_index < len(steps):
        return "executor"
    return "finalize"


# ---------------------------------------------------------------------------
# Build Vietnamese confirmation from completed step results
# ---------------------------------------------------------------------------

def build_execution_summary(state: State) -> AIMessage:
    raw_past_steps = state.get("past_steps") or []
    latest_by_step: dict[str, dict[str, Any]] = {}
    ordered_step_ids: list[str] = []
    for step in raw_past_steps:
        step_id = str(step.get("step_id") or "")
        if not step_id:
            continue
        if step_id not in latest_by_step:
            ordered_step_ids.append(step_id)
        latest_by_step[step_id] = step

    past_steps = [latest_by_step[step_id] for step_id in ordered_step_ids if step_id in latest_by_step]
    if not past_steps:
        return AIMessage(content="Đã hoàn thành các bước xử lý.")

    lines = []
    for step in past_steps:
        summary = str(step.get("summary") or "").strip()
        if not summary:
            continue
        status = str(step.get("status") or "")
        if status == "error":
            lines.append(f"❌ {summary}")
        elif status == "success":
            lines.append(f"✅ {summary}")
        elif status == "empty":
            lines.append(f"⚠️ {summary}")
        else:
            lines.append(summary)

    last_success = next(
        (step for step in reversed(past_steps) if str(step.get("status") or "") == "success"),
        None,
    )
    if last_success:
        output = last_success.get("output")
        if isinstance(output, list) and output and all(isinstance(row, dict) for row in output):
            lines.extend(_format_result_rows(output))

    return AIMessage(content="\n".join(lines))
