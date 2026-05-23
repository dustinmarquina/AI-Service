import json
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.orchestrator.llm import get_classifier_llm
from .state import State

logger = logging.getLogger(__name__)

Route = Literal["sql_agent", "chat", "execute", "finalize", "clarify"]

# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------

DB_SCHEMA = """
TABLE transactions (
    id UUID, amount NUMERIC(19,2), type VARCHAR(50),
    category_id UUID, category_name VARCHAR(255), description VARCHAR(255),
    wallet_id UUID, user_id UUID NOT NULL,
    transaction_date TIMESTAMPTZ, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ,
    image_url VARCHAR(1024), external_transaction_id VARCHAR(255)
)
TABLE wallets (
    id UUID, name VARCHAR(255), balance NUMERIC(19,2), currency VARCHAR(3),
    wallet_type VARCHAR(50), provider VARCHAR(255), provider_account_id VARCHAR(255),
    user_id UUID NOT NULL, active BOOLEAN, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ
)
TABLE categories (
    id UUID, name VARCHAR(255), icon VARCHAR(255), user_id UUID NOT NULL,
    budget_limit NUMERIC(19,2), period VARCHAR(50), active BOOLEAN,
    created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ, color VARCHAR(32)
)
""".strip()

TOOLS_SCHEMA = """
TOOL get_user_id()
  Resolves the current user's UUID from the bearer token.
  Returns: {"status": "success", "user_id": "<uuid>"}
  Always use this as the first step before any SQL query.

TOOL create_transaction(amount: str, description: str, wallet_id: str, category_name: str?)
  Records an EXPENSE transaction. Token is injected automatically — do not include it.

TOOL create_category(name: str, icon: str?, budget_limit: str?, period: str?)
  Creates a new spending category. Token is injected automatically — do not include it.
""".strip()

# ---------------------------------------------------------------------------
# Bootstrap step constant
# ---------------------------------------------------------------------------

GET_USER_ID_STEP = {
    "id": "s0",
    "type": "tool",
    "name": "get_user_id",
    "args": {},
    "write_to_state": "user_id",
}

# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = f"""You are a planner for a Vietnamese personal finance assistant.

Database schema:
{DB_SCHEMA}

Available tools:
{TOOLS_SCHEMA}

Given the user message and conversation history, output ONLY valid JSON — no markdown.

────────────────────────────────────────
CASE 1 — pure conversation, read/query answer, or simple single-step write (no DB lookup needed):
{{"route": "chat"|"sql_agent"|"finalize"|"clarify", "reason": "..."}}

CASE 2 — any action that requires reading the database (sql query OR tool that needs wallet_id/category_id):
{{
  "route": "execute",
  "reason": "...",
  "steps": [
    {{
      "id": "s0",
      "type": "tool",
      "name": "get_user_id",
      "args": {{}},
      "write_to_state": "user_id"
    }},
    ... your steps here (sql or tool) ...
  ]
}}
────────────────────────────────────────

Route rules:
- "chat"     : user wants to CREATE a transaction or category and NO db lookup is needed
- "sql_agent": use this for READ/query/summarise/reporting requests
- "execute"  : use this when user wants to write but needs wallet_id, category_id, or any DB value first
- "finalize" : pure conversation, no finance action
- "clarify"  : genuinely ambiguous
- For finance requests that omit an owner phrase like "của tôi", assume they refer to the authenticated current user by default.
- Do not ask the user to restate ownership for bare requests like "lần chi tiêu gần nhất", "tổng chi tuần vừa rồi", or "số dư hiện tại".

Step rules:
- ALWAYS start steps with id="s0" get_user_id as the FIRST step.
- After s0, use sql steps for DB reads and tool steps for writes.
- CRITICAL: if a tool step uses a placeholder like "$s1.id", then a step with id="s1" MUST
  exist earlier in the steps list. Never reference a step that isn't defined.
- Placeholder "$stepId.field" resolves to that field from the step result.
- Scope all SQL to the current user: WHERE user_id = :user_id

Common execute patterns:

Read query:
[s0: get_user_id, s1: sql "SELECT SUM(amount) FROM transactions WHERE user_id = :user_id AND ..."]

Write needing wallet:
[s0: get_user_id, s1: sql "SELECT id, name FROM wallets WHERE user_id = :user_id AND active = true", s2: tool create_transaction with wallet_id="$s1.id"]

Write needing category:
[s0: get_user_id, s1: sql "SELECT id FROM categories WHERE user_id = :user_id AND name ILIKE '%name%'", s2: tool create_transaction with category_id="$s1.id"]

Create category then record transaction:
[s0: get_user_id, s1: tool create_category, s2: sql fetch wallet, s3: tool create_transaction]
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLARIFY_RESPONSE = (
    "Mình chưa hiểu rõ ý bạn. "
    "Bạn muốn ghi một khoản chi tiêu, xem báo cáo, hay cần gì khác?"
)

_PLACEHOLDER_RE = re.compile(r"\$(\w+)\.(\w+)")

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_classifier_llm()
    return _llm


def _ensure_get_user_id_first(steps: list[dict]) -> list[dict]:
    """Guarantee s0/get_user_id is always the first step."""
    has_bootstrap = any(
        s.get("name") == "get_user_id" or s.get("id") == "s0"
        for s in steps
    )
    if not has_bootstrap:
        logger.warning("planner: get_user_id step missing, prepending automatically")
        return [GET_USER_ID_STEP] + steps

    bootstrap = next(s for s in steps if s.get("name") == "get_user_id" or s.get("id") == "s0")
    rest = [s for s in steps if s is not bootstrap]
    return [bootstrap] + rest


def _validate_execute_steps(steps: list[dict]) -> tuple[list[dict], bool]:
    """
    Sanitize and validate execute steps:
    - Ensure get_user_id is first
    - Drop steps with missing type/name
    - Verify every $stepId placeholder references a step that actually exists in the plan
    """
    if not isinstance(steps, list):
        return [GET_USER_ID_STEP], False

    sanitized = _ensure_get_user_id_first([s for s in steps if isinstance(s, dict)])
    actionable: list[dict] = []

    for step in sanitized[1:]:
        step_type = str(step.get("type") or "").strip().lower()

        if step_type == "sql":
            if step.get("description") or step.get("query_hint"):
                actionable.append(step)
            else:
                logger.warning("planner: dropping empty sql step: %s", step)

        elif step_type == "tool":
            if step.get("name"):
                actionable.append(step)
            else:
                logger.warning("planner: dropping tool step without name: %s", step)

        else:
            logger.warning("planner: dropping step with unknown type '%s': %s", step_type, step)

    if not actionable:
        return [GET_USER_ID_STEP], False

    all_steps = [sanitized[0]] + actionable
    defined_ids = {s.get("id") for s in all_steps}

    # Check every placeholder $sN.field has a matching step id in the plan
    broken: list[str] = []
    for step in actionable:
        args = step.get("args") or {}
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            for m in _PLACEHOLDER_RE.finditer(value):
                ref_id = m.group(1)
                if ref_id not in defined_ids:
                    broken.append(
                        f"step '{step.get('id')}' arg '{key}' "
                        f"references undefined step '${ref_id}'"
                    )

    if broken:
        logger.error(
            "planner: broken placeholder references — falling back to clarify:\n  %s",
            "\n  ".join(broken),
        )
        return [GET_USER_ID_STEP], False

    return all_steps, True


def _parse_planner_output(text: str) -> dict:
    try:
        clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        data = json.loads(clean)
        route = data.get("route", "clarify")
        if route not in ("chat", "sql_agent", "execute", "finalize", "clarify"):
            logger.warning("planner: unknown route '%s'", route)
            return {"route": "clarify", "reason": "unknown route"}

        if route == "execute":
            steps, is_valid = _validate_execute_steps(data.get("steps", []))
            if not is_valid:
                return {"route": "clarify", "reason": "invalid execute steps"}
            data["steps"] = steps

        return data
    except (json.JSONDecodeError, AttributeError):
        logger.warning("planner: failed to parse: %s", text[:200])
        return {"route": "clarify", "reason": "parse error"}


def _fallback_route(messages: list) -> dict:
    latest_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    text = str(latest_human.content if latest_human else "").strip().lower()
    normalized = " ".join(text.split())

    if not normalized:
        return {"route": "clarify", "reason": "empty message"}

    query_markers = (
        "tổng", "bao nhiêu", "thống kê", "báo cáo", "liệt kê",
        "xem", "còn dư", "số dư", "vừa rồi", "tháng", "tuần",
        "hôm nay", "hôm qua", "danh mục", "gần nhất", "mới nhất",
        "cuối cùng", "chi tiêu gần nhất", "giao dịch gần nhất",
    )
    create_markers = ("tạo category", "tao category", "tạo danh mục", "tao danh muc")
    amount_pattern = r"\b\d+(?:[.,]\d+)?\s*(?:k|nghìn|ngàn|triệu|tr|đ|dong|vnd)\b"

    if any(m in normalized for m in create_markers):
        return {
            "route": "execute",
            "reason": "fallback: category creation",
            "steps": [
                GET_USER_ID_STEP,
                {
                    "id": "s1", "type": "tool", "name": "create_category",
                    "args": {"name": re.sub("|".join(create_markers), "", normalized).strip()},
                },
            ],
        }

    if any(m in normalized for m in query_markers):
        return {
            "route": "sql_agent",
            "reason": "fallback: reporting query",
        }

    if re.search(amount_pattern, normalized):
        amount_match = re.search(amount_pattern, normalized)
        desc = re.sub(amount_pattern, "", normalized).strip()
        return {
            "route": "execute",
            "reason": "fallback: transaction needs wallet lookup",
            "steps": [
                GET_USER_ID_STEP,
                {
                    "id": "s1", "type": "sql",
                    "description": "fetch user active wallets",
                    "query_hint": "SELECT id, name, balance, currency FROM wallets WHERE user_id = :user_id AND active = true",
                },
                {
                    "id": "s2", "type": "tool", "name": "create_transaction",
                    "args": {
                        "amount": amount_match.group(0),
                        "description": desc,
                        "wallet_id": "$s1.id",
                    },
                },
            ],
        }

    if normalized in {"ok", "oke", "cảm ơn", "cam on", "hello", "hi"}:
        return {"route": "finalize", "reason": "fallback: casual chat"}

    return {"route": "clarify", "reason": "fallback: undetermined"}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def planner_node(state: State) -> dict:
    messages = list(state.get("messages", []))
    history = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
    ][-6:]

    llm = _get_llm()
    try:
        response = await llm.ainvoke([SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + history)
        raw = response.content if hasattr(response, "content") else str(response)
        result = _parse_planner_output(raw)
    except Exception as exc:
        logger.warning("planner: LLM failed, fallback. error=%s", exc)
        result = _fallback_route(history)

    if result.get("reason") in {"parse error", "invalid execute steps"}:
        result = _fallback_route(history)

    logger.info(
        "planner: route=%s reason=%s steps=%d",
        result.get("route"), result.get("reason"), len(result.get("steps", [])),
    )

    update: dict = {"route": result["route"]}
    if result["route"] == "execute":
        update["steps"] = result.get("steps", [])
        update["step_index"] = 0
        update["step_results"] = {}

    return update


def route_after_planner(state: State) -> Route:
    route = state.get("route", "clarify")
    if route not in ("chat", "sql_agent", "execute", "finalize", "clarify"):
        logger.warning("route_after_planner: unexpected '%s', clarifying", route)
        return "clarify"
    return route


async def clarify_node(state: State) -> dict:
    return {"messages": [AIMessage(content=_CLARIFY_RESPONSE)]}
