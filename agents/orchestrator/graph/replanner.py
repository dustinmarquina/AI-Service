import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.orchestrator.candidate_binding import candidate_value, select_candidate_binding
from agents.orchestrator.llm import get_classifier_llm
from .state import State

logger = logging.getLogger(__name__)

Route = Literal["execute", "finalize", "clarify"]

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_classifier_llm()
    return _llm


def _default_max_replans(state: State) -> int:
    value = state.get("max_replan_attempts")
    if isinstance(value, int) and value >= 0:
        return value
    return 2


def _build_replanner_prompt(state: State) -> str:
    steps = state.get("steps") or []
    step_index = state.get("step_index") or 0
    past_steps = state.get("past_steps") or []
    return (
        "You are a generic replanner for a plan-and-execute agent.\n"
        "Given the original conversation, the full current plan, the next step index, and past step results,\n"
        "decide whether to continue execution, replace the remaining steps, finalize, or ask a clarifying question.\n"
        "Output ONLY valid JSON.\n\n"
        "Allowed routes:\n"
        '- {"route":"execute","reason":"...","steps":[...]}\n'
        '- {"route":"finalize","reason":"...","message":"..."}\n'
        '- {"route":"clarify","reason":"...","message":"..."}\n\n'
        f"Current steps: {json.dumps(steps, ensure_ascii=False)}\n"
        f"Current step_index: {step_index}\n"
        f"Past steps: {json.dumps(past_steps, ensure_ascii=False)}"
    )

def _parse_replanner_output(text: str) -> dict:
    clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
    data = json.loads(clean)
    route = data.get("route")
    if route not in ("execute", "finalize", "clarify"):
        raise ValueError(f"unsupported route: {route}")
    return data


def _steps_have_supported_placeholders(steps: list[dict]) -> bool:
    placeholder_re = re.compile(r"^\$(\w+)\.(\w+)$")
    defined_ids: set[str] = set()
    for step in steps:
        step_id = step.get("id")
        if step_id:
            defined_ids.add(step_id)
        for value in (step.get("args") or {}).values():
            if not isinstance(value, str) or not value.startswith("$"):
                continue
            match = placeholder_re.match(value.strip())
            if not match or match.group(1) not in defined_ids:
                return False
    return True


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _candidate_arg_names(step: dict) -> list[str]:
    names: set[str] = set()
    args = step.get("args") or {}
    if isinstance(args, dict):
        for key, value in args.items():
            if key.endswith("_id") and not value:
                names.add(key)

    step_text = " ".join(
        str(step.get(key) or "")
        for key in ("reasoning", "description", "name")
    )
    names.update(re.findall(r"\b([a-zA-Z][a-zA-Z0-9_]*_id)\b", step_text))
    return sorted(names)


def _arg_base_name(arg_name: str) -> str:
    return arg_name[:-3] if arg_name.endswith("_id") else arg_name


def _step_candidate_rows(step: dict, step_results: dict[str, object]) -> list[dict] | None:
    result = step_results.get(str(step.get("id") or ""))
    if isinstance(result, list) and result and all(isinstance(row, dict) for row in result):
        return result
    if isinstance(result, dict):
        return [result]
    return None


def _relevance_score_for_arg(step: dict, arg_name: str, rows: list[dict]) -> int:
    base_name = _arg_base_name(arg_name).lower()
    metadata = " ".join(
        str(step.get(key) or "").lower()
        for key in ("query_hint", "description", "reasoning")
    )
    score = 0
    if arg_name.lower() in metadata:
        score += 100
    if base_name and base_name in metadata:
        score += 80
    plural = f"{base_name}s"
    if base_name and plural in metadata:
        score += 40

    if rows:
        first_row = rows[0]
        if isinstance(first_row, dict):
            if arg_name in first_row:
                score += 25
            camel_name = _snake_to_camel(arg_name)
            if camel_name in first_row:
                score += 20
            if "id" in first_row:
                score += 10
    return score


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    if not parts:
        return value
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _find_candidate_rows_for_arg(steps: list[dict], step_results: dict[str, object], arg_name: str) -> list[dict] | None:
    best_rows: list[dict] | None = None
    best_score = 0
    sql_steps_with_rows = 0
    fallback_rows: list[dict] | None = None

    for step in steps:
        if str(step.get("type") or "").lower() != "sql":
            continue
        rows = _step_candidate_rows(step, step_results)
        if not rows:
            continue
        sql_steps_with_rows += 1
        fallback_rows = rows
        score = _relevance_score_for_arg(step, arg_name, rows)
        if score > best_score:
            best_score = score
            best_rows = rows

    if best_rows is not None and best_score > 0:
        return best_rows
    if sql_steps_with_rows == 1:
        return fallback_rows
    return None


async def _llm_bind_candidate_arg(
    *,
    current_step: dict[str, Any],
    arg_name: str,
    rows: list[dict[str, Any]],
    user_text: str,
) -> dict | None:
    llm = _get_llm()
    decision = await select_candidate_binding(
        llm,
        user_text=user_text,
        current_context=current_step,
        arg_name=arg_name,
        rows=rows,
    )
    if decision is None:
        logger.warning("replanner: failed candidate binding for %s", arg_name)
        return None
    return decision


async def _maybe_bind_missing_id_arg(state: State) -> dict | None:
    steps = list(state.get("steps") or [])
    step_index = int(state.get("step_index") or 0)
    step_results = dict(state.get("step_results") or {})
    replan_attempts = int(state.get("replan_attempts") or 0)

    if not steps or step_index >= len(steps):
        return None

    current_step = dict(steps[step_index] or {})
    if str(current_step.get("type") or "").lower() != "tool":
        return None

    args = dict(current_step.get("args") or {})
    user_text = _latest_user_text(list(state.get("messages") or []))
    updated = False

    for arg_name in _candidate_arg_names(current_step):
        if args.get(arg_name):
            continue
        rows = _find_candidate_rows_for_arg(steps[:step_index], step_results, arg_name)
        if not rows:
            continue

        if len(rows) == 1:
            selected_value = candidate_value(rows[0], arg_name)
            if not selected_value:
                continue
        else:
            decision = await _llm_bind_candidate_arg(
                current_step=current_step,
                arg_name=arg_name,
                rows=rows,
                user_text=user_text,
            )
            if decision is None:
                continue
            if decision["route"] == "clarify":
                return {
                    "route": "clarify",
                    "replan_attempts": replan_attempts,
                    "messages": [AIMessage(content=decision["message"])],
                }
            selected_value = decision["value"]

        args[arg_name] = selected_value
        updated = True

    if not updated:
        return None

    current_step["args"] = args
    new_steps = list(steps)
    new_steps[step_index] = current_step
    return {
        "route": "execute",
        "steps": new_steps,
        "step_index": step_index,
        "step_results": step_results,
        "replan_attempts": replan_attempts,
    }


async def replanner_node(state: State) -> dict:
    steps = list(state.get("steps") or [])
    step_index = int(state.get("step_index") or 0)
    past_steps = list(state.get("past_steps") or [])
    step_results = dict(state.get("step_results") or {})
    replan_attempts = int(state.get("replan_attempts") or 0)
    max_replan_attempts = _default_max_replans(state)

    bound_candidate_update = await _maybe_bind_missing_id_arg(state)
    if bound_candidate_update is not None:
        return bound_candidate_update

    if step_index < len(steps):
        return {"route": "execute", "replan_attempts": replan_attempts}

    if not past_steps:
        return {
            "route": "clarify",
            "replan_attempts": replan_attempts,
            "messages": [AIMessage(content="Mình cần thêm thông tin để tiếp tục.")],
        }

    last_step = past_steps[-1]
    last_status = str(last_step.get("status") or "")

    if last_status == "success":
        return {"route": "finalize", "replan_attempts": replan_attempts}

    if replan_attempts >= max_replan_attempts:
        return {
            "route": "clarify",
            "replan_attempts": replan_attempts,
            "messages": [
                AIMessage(
                    content=f"Không thể tiếp tục sau {replan_attempts} lần thử lại: {last_step.get('summary', 'lỗi không xác định')}"
                )
            ],
        }

    llm = _get_llm()
    history = [
        message for message in list(state.get("messages") or [])
        if isinstance(message, (HumanMessage, AIMessage))
    ][-8:]
    try:
        response = await llm.ainvoke([SystemMessage(content=_build_replanner_prompt(state))] + history)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_replanner_output(raw)
    except Exception as exc:
        logger.warning("replanner: failed to parse model output: %s", exc)
        return {
            "route": "clarify",
            "replan_attempts": replan_attempts,
            "messages": [AIMessage(content=f"Mình cần làm rõ trước khi thử lại: {last_step.get('summary', 'lỗi không xác định')}")],
        }

    if parsed["route"] == "execute":
        new_steps = parsed.get("steps", [])
        if not _steps_have_supported_placeholders(new_steps):
            return {
                "route": "clarify",
                "replan_attempts": replan_attempts,
                "messages": [AIMessage(content="I need to clarify the next step before retrying.")],
            }
        return {
            "route": "execute",
            "steps": new_steps,
            "step_index": 0,
            "step_results": {},
            "replan_attempts": replan_attempts + 1,
        }

    if parsed["route"] == "finalize":
        return {
            "route": "finalize",
            "replan_attempts": replan_attempts,
            "messages": [AIMessage(content=str(parsed.get("message") or "Done."))],
        }

    return {
        "route": "clarify",
        "replan_attempts": replan_attempts,
        "messages": [AIMessage(content=str(parsed.get("message") or "Please clarify."))],
    }


def route_after_replanner(state: State) -> str:
    route = state.get("route", "clarify")
    if route not in ("execute", "finalize", "clarify"):
        logger.warning("route_after_replanner: unexpected '%s', clarifying", route)
        return "clarify"
    if route == "execute":
        return "executor"
    return route
