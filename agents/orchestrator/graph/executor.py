import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .sql_agent import get_sql_agent
from .state import State

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder resolution
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"^\$(\w+)\.(\w+)$")


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


# ---------------------------------------------------------------------------
# SQL result extraction
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _extract_sql_rows(result: dict) -> list[dict]:
    """
    Extract structured rows from the SQL agent result.

    The SQL agent returns a Vietnamese natural-language answer. We parse it
    looking for wallet/category rows. Supports two formats:

    Format A — one UUID per line with a name:
        "1. Tiền mặt (uuid-...)\n2. VCB (uuid-...)"

    Format B — JSON array in the message content.
    """
    messages = result.get("messages", [])
    content = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            content = str(m.content).strip()
            break

    # Try JSON first
    try:
        import json
        # find a JSON array anywhere in the content
        array_match = re.search(r"\[.*\]", content, re.DOTALL)
        if array_match:
            data = json.loads(array_match.group(0))
            if isinstance(data, list) and all(isinstance(r, dict) for r in data):
                return data
    except Exception:
        pass

    # Extract UUIDs with optional adjacent name
    uuids = _UUID_RE.findall(content)
    if not uuids:
        return []

    if len(uuids) == 1:
        # Single result — no need for user selection
        return [{"id": uuids[0], "value": uuids[0], "name": "", "raw": content}]

    # Multiple UUIDs — try to pair each with a name from the same line
    rows = []
    for uuid in uuids:
        # Find the line containing this UUID
        for line in content.splitlines():
            if uuid.lower() in line.lower():
                # Strip the UUID and numbering to get the name
                name = _UUID_RE.sub("", line)
                name = re.sub(r"^\s*\d+[\.\)]\s*", "", name)  # remove "1. " prefix
                name = name.strip(" ()-:,")
                rows.append({"id": uuid, "name": name or uuid[:8], "raw": line.strip()})
                break
        else:
            rows.append({"id": uuid, "name": uuid[:8], "raw": uuid})

    return rows


def _build_wallet_prompt(rows: list[dict]) -> str:
    lines = ["Bạn có các ví sau, chọn ví muốn dùng:"]
    for i, row in enumerate(rows, 1):
        name = row.get("name") or row.get("id", "")[:8]
        balance = row.get("balance", "")
        currency = row.get("currency", "")
        detail = f" — {balance:,} {currency}".strip() if balance else ""
        lines.append(f"  {i}. {name}{detail}")
    lines.append("\nNhập số thứ tự hoặc tên ví:")
    return "\n".join(lines)


def _resolve_wallet_selection(selection: str, rows: list[dict]) -> dict | None:
    """Match user reply to a wallet row by index or name. Returns the matched row or None."""
    selection = selection.strip()

    # By index
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(rows):
            return rows[idx]
        return None

    # By name (case-insensitive substring)
    sel_lower = selection.lower()
    for row in rows:
        name = str(row.get("name", "")).lower()
        if sel_lower in name or name in sel_lower:
            return rows[rows.index(row)]

    return None


# ---------------------------------------------------------------------------
# Executor node
# ---------------------------------------------------------------------------

async def executor_node(state: State) -> dict:
    steps: list[dict] = state.get("steps") or []
    step_index: int = state.get("step_index") or 0
    step_results: dict = dict(state.get("step_results") or {})

    if step_index >= len(steps):
        logger.warning("executor: step_index=%d >= len(steps)=%d", step_index, len(steps))
        return {"step_index": step_index}

    current = steps[step_index]
    step_id = current["id"]
    step_type = current["type"]

    logger.info(
        "executor: step %d/%d id=%s type=%s",
        step_index + 1, len(steps), step_id, step_type,
    )

    # ── SQL step ──────────────────────────────────────────────────────────────
    if step_type == "sql":
        query_hint = current.get("query_hint", "")
        description = current.get("description", "")
        needs_selection = current.get("needs_user_selection", False)

        user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()
        if user_id:
            query_hint = query_hint.replace(":user_id", f"'{user_id}'")

        prompt = (
            f"{description}\n\nSQL hint:\n{query_hint}" if query_hint else description
        )

        sql_agent = get_sql_agent()
        result = await sql_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        rows = _extract_sql_rows(result)

        if not rows:
            logger.warning("executor: sql step %s returned no rows", step_id)
            return {
                "messages": [AIMessage(content="Không tìm thấy dữ liệu phù hợp.")],
                "step_index": len(steps),   # skip remaining steps
                "step_results": step_results,
            }

        # Single row — no selection needed, continue automatically
        if len(rows) == 1 and not needs_selection:
            step_results[step_id] = rows[0]
            logger.info("executor: step %s single row, auto-selected: %s", step_id, rows[0])
            return {
                "step_index": step_index + 1,
                "step_results": step_results,
            }

        # Multiple rows (or forced selection) — interrupt and ask the user
        wallet_prompt = _build_wallet_prompt(rows)

        # Store rows in step_results under a staging key so we can resolve after resume
        step_results[f"{step_id}__candidates"] = rows

        logger.info("executor: interrupting for user wallet selection, %d options", len(rows))

        # interrupt() pauses the graph and returns wallet_prompt to the caller.
        # Execution resumes when Command(resume=<user_reply>) is sent.
        selection: str = interrupt(wallet_prompt)

        # ── Resumed ──────────────────────────────────────────────────────────
        candidates: list[dict] = step_results.get(f"{step_id}__candidates", [])
        matched = _resolve_wallet_selection(selection, candidates)

        if matched is None:
            # Could not match — re-interrupt with a clearer prompt
            retry_prompt = (
                f"Mình chưa nhận ra ví đó. {_build_wallet_prompt(candidates)}"
            )
            selection = interrupt(retry_prompt)
            matched = _resolve_wallet_selection(selection, candidates)

        if matched is None:
            return {
                "messages": [AIMessage(content="Không xác định được ví. Vui lòng thử lại.")],
                "step_index": len(steps),
                "step_results": step_results,
            }

        step_results[step_id] = matched
        logger.info("executor: user selected wallet: %s", matched)

        return {
            "step_index": step_index + 1,
            "step_results": step_results,
        }

    # ── Tool step ─────────────────────────────────────────────────────────────
    elif step_type == "tool":
        tool_name = current.get("name", "")
        raw_args = current.get("args", {})

        try:
            resolved_args = _resolve_args(raw_args, step_results)
        except ValueError as exc:
            logger.error("executor: placeholder resolution failed: %s", exc)
            return {
                "messages": [AIMessage(content=f"Lỗi xử lý bước {step_id}: {exc}")],
                "step_index": len(steps),
                "step_results": step_results,
            }

        # Inject runtime credentials
        runtime_token = str(state.get("token") or os.getenv("TRANSACTION_API_TOKEN", "")).strip()
        runtime_user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()
        if runtime_token:
            resolved_args.setdefault("token", runtime_token)
        if runtime_user_id:
            resolved_args.setdefault("user_id", runtime_user_id)

        logger.info("executor: calling tool '%s' args=%s", tool_name, {
            k: (v[:8] + "...") if k == "token" and isinstance(v, str) else v
            for k, v in resolved_args.items()
        })

        from .tool_registry import call_mcp_tool
        tool_result = await call_mcp_tool(tool_name, resolved_args)
        step_results[step_id] = tool_result

        logger.info("executor: step %s tool result=%s", step_id, str(tool_result)[:200])

        return {
            "step_index": step_index + 1,
            "step_results": step_results,
        }

    else:
        logger.error("executor: unknown step type '%s'", step_type)
        return {
            "step_index": step_index + 1,
            "step_results": step_results,
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
    step_results = state.get("step_results") or {}
    steps = state.get("steps") or []
    tool_steps = [s for s in steps if s["type"] == "tool"]

    if not tool_steps:
        return AIMessage(content="Đã hoàn thành các bước xử lý.")

    lines = []
    for step in tool_steps:
        result = step_results.get(step["id"], {})
        tool_name = step.get("name", "")
        args = step.get("args", {})

        if tool_name == "create_transaction":
            status = result.get("status", "unknown")
            data = result.get("data", {})
            if status == "success":
                amount = data.get("amount", args.get("amount", ""))
                desc = data.get("description", args.get("description", ""))
                lines.append(f"✅ Đã ghi nhận chi tiêu **{int(amount):,} ₫** cho **{desc}**.")
            else:
                lines.append(f"❌ Ghi giao dịch thất bại: {result.get('message', 'lỗi không xác định')}")

        elif tool_name == "create_category":
            status = result.get("status", "unknown")
            name = args.get("name", "")
            if status == "success":
                lines.append(f"✅ Đã tạo danh mục **{name}**.")
            else:
                lines.append(f"❌ Tạo danh mục thất bại: {result.get('message', 'lỗi không xác định')}")

        else:
            lines.append(f"✅ Bước {step['id']} hoàn thành.")

    return AIMessage(content="\n".join(lines))