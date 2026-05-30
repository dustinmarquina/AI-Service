import json
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.orchestrator.llm import get_classifier_llm
from .state import State
from .tool_registry import describe_tools

logger = logging.getLogger(__name__)

Route = Literal["sql_agent", "execute", "finalize", "clarify"]

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

def _build_planner_system_prompt() -> str:
        tools_schema = describe_tools()
        return f"""You are a planner for a Vietnamese personal finance assistant.

Database schema:
{DB_SCHEMA}

Available tools:
{tools_schema}

Given the user message and conversation history, output ONLY valid JSON — no markdown.

────────────────────────────────────────
CASE 1 — pure conversation, read/query answer, or simple single-step write (no DB lookup needed):
{{"route": "sql_agent"|"finalize"|"clarify", "reason": "..."}}

CASE 2 — financial advice or any action that requires reading the database (sql query OR tool that needs wallet_id/category_id):
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
- "execute"  : use this when user wants to write but needs wallet_id, category_id, or any DB value first
- "execute"  : use this when user wants advice, wants to write, or needs wallet_id/category_id/any DB value first
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

Advice / financial-health query:
[s0: get_user_id, s1: tool get_wallet_summary]
"""


PLANNER_SYSTEM_PROMPT = _build_planner_system_prompt()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLARIFY_RESPONSE = (
    "Mình chưa hiểu rõ ý bạn. "
    "Bạn muốn ghi một khoản chi tiêu, xem báo cáo, hay cần gì khác?"
)

_PLACEHOLDER_RE = re.compile(r"\$(\w+)\.(\w+)")
_ADVICE_RE = re.compile(
    r"(tư vấn|tu van|advice|financial tips|financial advice|gợi ý|goi y|"
    r"khuyên|khuyen|nên|nen|sức khỏe tài chính|suc khoe tai chinh)",
    re.IGNORECASE,
)

_CREATE_CATEGORY_RE = re.compile(r"\b(tạo category|tao category|tạo danh mục|tao danh muc)\b", re.IGNORECASE)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_classifier_llm()
    return _llm


def _is_advice_request(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    return bool(normalized and _ADVICE_RE.search(normalized))


def _build_advice_execute_plan() -> dict:
    return {
        "route": "execute",
        "reason": "advice merged into execute",
        "steps": [
            GET_USER_ID_STEP,
            {
                "id": "s1",
                "type": "tool",
                "name": "get_wallet_summary",
                "args": {},
            },
        ],
    }


def _build_create_category_execute_plan(category_name: str) -> dict:
    name = " ".join(str(category_name or "").strip().split())
    if not name:
        return {
            "route": "clarify",
            "reason": "missing category name",
        }

    return {
        "route": "execute",
        "reason": "category creation merged into execute",
        "steps": [
            GET_USER_ID_STEP,
            {
                "id": "s1",
                "type": "tool",
                "name": "create_category",
                "args": {"name": name},
            },
        ],
    }


def _extract_category_name_from_history(messages: list) -> str:
    human_messages = [
        m for m in messages
        if isinstance(m, HumanMessage)
        and str(getattr(m, "content", "")).strip()
    ]
    latest_text = " ".join(str(human_messages[-1].content).strip().lower().split())
    if _CREATE_CATEGORY_RE.search(latest_text):
        remainder = _CREATE_CATEGORY_RE.sub("", str(human_messages[-1].content), count=1).strip(" ,.:;-_")
        if remainder:
            return " ".join(remainder.split())

    if len(human_messages) < 2:
        return ""

    if not _CREATE_CATEGORY_RE.search(latest_text):
        return ""

    for prior in reversed(human_messages[:-1]):
        candidate = " ".join(str(prior.content).strip().split())
        candidate_lower = candidate.lower()
        if not candidate or _CREATE_CATEGORY_RE.search(candidate_lower):
            continue
        if len(candidate) < 2:
            continue
        if candidate_lower in {"ok", "oke", "cảm ơn", "cam on", "thanks", "thank you"}:
            continue
        return candidate

    return ""


def _build_category_clarify_response(messages: list) -> str:
    category_name = _extract_category_name_from_history(messages)
    if category_name:
        return (
            f"Mình sẽ tạo category '{category_name}'. "
            "Nếu bạn muốn, hãy gửi thêm icon hoặc hạn mức chi tiêu."
        )

    return "Bạn muốn tạo category tên gì?"


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


def _parse_planner_output(text: str, latest_human_text: str = "") -> dict:
    try:
        clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        data = json.loads(clean)
        route = data.get("route", "clarify")
        if route not in ("sql_agent", "execute", "finalize", "clarify"):
            logger.warning("planner: unknown route '%s'", route)
            return {"route": "clarify", "reason": "unknown route"}

        if route == "chat":
            return _build_advice_execute_plan() if _is_advice_request(latest_human_text) else {
                "route": "clarify",
                "reason": "chat removed from planner",
            }

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

    if _is_advice_request(normalized):
        return _build_advice_execute_plan()

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
        response = await llm.ainvoke([SystemMessage(content=_build_planner_system_prompt())] + history)
        raw = response.content if hasattr(response, "content") else str(response)
        latest_human_text = str(next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""))
        result = _parse_planner_output(raw, latest_human_text)
    except Exception as exc:
        logger.warning("planner: LLM failed, fallback. error=%s", exc)
        result = _fallback_route(history)

    if result.get("route") == "clarify":
        latest_human_text = str(next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""))
        if _CREATE_CATEGORY_RE.search(" ".join(latest_human_text.strip().lower().split())):
            category_name = _extract_category_name_from_history(history)
            if category_name:
                result = _build_create_category_execute_plan(category_name)

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
    if route not in ("sql_agent", "execute", "finalize", "clarify"):
        logger.warning("route_after_planner: unexpected '%s', clarifying", route)
        return "clarify"
    return route


async def clarify_node(state: State) -> dict:
    messages = list(state.get("messages", []))
    latest_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    latest_text = str(getattr(latest_human, "content", "")).strip().lower()

    if _CREATE_CATEGORY_RE.search(" ".join(latest_text.split())):
        return {"messages": [AIMessage(content=_build_category_clarify_response(messages))]}

    return {"messages": [AIMessage(content=_CLARIFY_RESPONSE)]}
