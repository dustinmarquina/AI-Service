import json
import logging
import re
import unicodedata
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.orchestrator.llm import get_classifier_llm
from .config import DEFAULT_CLARIFY_RESPONSE
from .state import State
from .tool_registry import describe_tools

logger = logging.getLogger(__name__)

Route = Literal["execute", "finalize", "clarify"]

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

GET_USER_ID_STEP = {
    "id": "s0",
    "type": "tool",
    "name": "get_user_id",
    "args": {},
    "write_to_state": "user_id",
}

_PLACEHOLDER_RE = re.compile(r"^\$(\w+)\.(\w+)$")
_AMOUNT_TOKEN_RE = re.compile(
    r"(?P<amount>\d+(?:[.,]\d+)?)\s*(?:k|ng[aà]n|ngh[iì]n|nghin|tr|tri[eê]u|m|vnd|vnđ|đ|d)\b|(?P<grouped>\d{1,3}(?:[.,]\d{3})+)\b",
    re.IGNORECASE,
)
_WALLET_HINT_RE = re.compile(r"\b(?:v[aà]o\s+v[ií]|v[ií])\s+(.+)$", re.IGNORECASE)
_GENERIC_TRANSACTION_DESCRIPTIONS = {
    "giao dich",
    "them giao dich",
    "tao giao dich",
    "them mot giao dich",
    "chi tieu",
    "them chi tieu",
    "tao chi tieu",
    "expense",
    "add expense",
    "transaction",
    "add transaction",
    "create transaction",
}
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_classifier_llm()
    return _llm


def _build_planner_system_prompt() -> str:
    tools_schema = describe_tools()
    return f"""You are a generic planner for a plan-and-execute agent.

You can produce plans using two step types:
- "tool": call an MCP tool by name with args
- "sql": run a read-only SQL lookup using query_hint

Available tools:
{tools_schema}

Available SQL data sources:
{DB_SCHEMA}

Return ONLY valid JSON with one of these shapes:
{{"route":"execute","reason":"...","steps":[...]}}
{{"route":"finalize","reason":"...","message":"..."}}
{{"route":"clarify","reason":"...","message":"..."}}

Planning rules:
- Use "execute" when the request needs one or more tool or sql steps.
- Use "finalize" only when you can answer directly without executing any step.
- Use "clarify" when the request has nothing to do with given tools or SQL schema, otherwise attempt to execute steps but the requirements are too ambiguous or complex to resolve with a single SQL lookup or tool call.
- Do not use "clarify" when the user explicitly commands an action that matches a tool's purpose, even there's missing fields.
- Do not generate a generic value for the missing fields, just ignore them and let the executor handle it, which may result in a clarifying question later, but that's fine.
- If ambiguity can be deferred to a later selection step or a tool can request follow-up input during execution, prefer "execute" over "clarify".
- SQL steps must be SELECT-only and use fields:
  {{"id":"sN","type":"sql","reasoning":"...","description":"...","query_hint":"SELECT ...","selection_mode":"none|required"}}
- Tool steps must use fields:
  {{"id":"sN","type":"tool","name":"tool_name","reasoning":"...","args":{{...}}}}
- Each step must include a "reasoning" field explaining why this step is necessary given the tool requirements and what is already known.
- Use "selection_mode":"required" only when later execution needs exactly one row chosen from multiple candidates. Otherwise use "selection_mode":"none" or omit it.
- If the user already provides a concrete identifier or name needed to disambiguate a later step, pass it through in tool args or SQL filters instead of planning a follow-up prompt.
- Placeholder references must point only to earlier steps, e.g. "$s1.id".
- Placeholder references must be exactly "$sN.field". Do not use array indexing, nested paths, or expressions like "$s1.items[0].id".
- When authenticated user context is needed, begin with:
  {json.dumps(GET_USER_ID_STEP)}
- Do not invent tools that are not listed.
- Do not use domain-specific shortcuts. Reason only from the request, tool signatures, SQL schema, and prior conversation.
"""


def _ensure_get_user_id_first(steps: list[dict]) -> list[dict]:
    has_bootstrap = any(
        s.get("name") == "get_user_id" or s.get("id") == "s0"
        for s in steps
    )
    if not has_bootstrap:
        return [GET_USER_ID_STEP] + steps

    bootstrap = next(s for s in steps if s.get("name") == "get_user_id" or s.get("id") == "s0")
    rest = [s for s in steps if s is not bootstrap]
    return [bootstrap] + rest


_COMPARATIVE_TOP1_SQL_RE = re.compile(
    r"^\s*SELECT\s+.+?\s+FROM\s+(wallets|categories|transactions)\s+.+\bORDER\s+BY\b.+\bLIMIT\s+1\b",
    re.IGNORECASE,
)

_TABLE_CANDIDATE_SELECT = {
    "wallets": "SELECT id, name, balance, currency FROM wallets",
    "categories": "SELECT id, name, icon, budget_limit, period FROM categories",
    "transactions": "SELECT id, description, amount, transaction_date, created_at FROM transactions",
}


def _query_references_step(step_id: str, step: dict) -> bool:
    args = step.get("args") or {}
    for value in args.values():
        if isinstance(value, str) and value.startswith(f"${step_id}."):
            return True
    return False


def _strip_dependency_placeholders(step_id: str, step: dict) -> dict:
    updated = dict(step)
    args = dict(updated.get("args") or {})
    changed = False
    for key, value in list(args.items()):
        if isinstance(value, str) and value.startswith(f"${step_id}.") and key.endswith("_id"):
            args.pop(key, None)
            changed = True
    if changed:
        updated["args"] = args
    return updated


def _extract_from_clause(query_hint: str) -> tuple[str, str] | None:
    match = re.search(r"\bFROM\s+(wallets|categories|transactions)\b(.*)$", query_hint, re.IGNORECASE)
    if not match:
        return None
    table_name = match.group(1).lower()
    remainder = match.group(2)
    remainder = re.sub(r"\bORDER\s+BY\b.+$", "", remainder, flags=re.IGNORECASE).strip()
    remainder = re.sub(r"\bLIMIT\s+\d+\b", "", remainder, flags=re.IGNORECASE).strip()
    return table_name, remainder


def _rewrite_dependency_lookup_steps(steps: list[dict]) -> list[dict]:
    rewritten = [dict(step) for step in steps]

    for index, step in enumerate(rewritten):
        if str(step.get("type") or "").lower() != "sql":
            continue
        query_hint = str(step.get("query_hint") or "").strip()
        step_id = str(step.get("id") or "")
        if not step_id or not _COMPARATIVE_TOP1_SQL_RE.match(query_hint):
            continue
        referenced = False
        for next_index in range(index + 1, len(rewritten)):
            next_step = rewritten[next_index]
            if str(next_step.get("type") or "").lower() != "tool":
                continue
            if not _query_references_step(step_id, next_step):
                continue
            referenced = True
            rewritten[next_index] = _strip_dependency_placeholders(step_id, next_step)

        if not referenced:
            continue

        extracted = _extract_from_clause(query_hint)
        if extracted is None:
            continue

        table_name, remainder = extracted
        query_prefix = _TABLE_CANDIDATE_SELECT.get(table_name)
        if not query_prefix:
            continue

        rebuilt_query = query_prefix
        if remainder:
            rebuilt_query = f"{rebuilt_query} {remainder}".strip()

        step["query_hint"] = rebuilt_query
        step["selection_mode"] = "none"
        step["description"] = str(step.get("description") or "").strip() or (
            f"Fetch candidate {table_name} rows so the later tool step can resolve the correct id."
        )

    return rewritten


def _validate_execute_steps(steps: list[dict]) -> tuple[list[dict], bool]:
    if not isinstance(steps, list):
        return [GET_USER_ID_STEP], False

    sanitized = _ensure_get_user_id_first([s for s in steps if isinstance(s, dict)])
    actionable: list[dict] = []

    for step in sanitized[1:]:
        step_type = str(step.get("type") or "").strip().lower()
        reasoning = str(step.get("reasoning") or "").strip()
        if not reasoning:
            logger.warning("planner: dropping step without reasoning: %s", step)
            continue

        if step_type == "sql":
            if step.get("query_hint"):
                actionable.append(dict(step))
            else:
                logger.warning("planner: dropping empty sql step: %s", step)
        elif step_type == "tool":
            if step.get("name"):
                actionable.append(dict(step))
            else:
                logger.warning("planner: dropping tool step without name: %s", step)
        else:
            logger.warning("planner: dropping step with unknown type '%s': %s", step_type, step)

    if not actionable:
        return [GET_USER_ID_STEP], False

    all_steps = _rewrite_dependency_lookup_steps([sanitized[0]] + actionable)
    defined_ids = {s.get("id") for s in all_steps}
    broken: list[str] = []

    for step in actionable:
        args = step.get("args") or {}
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            if value.startswith("$"):
                match = _PLACEHOLDER_RE.match(value.strip())
                if not match:
                    broken.append(
                        f"step '{step.get('id')}' arg '{key}' uses unsupported placeholder syntax '{value}'"
                    )
                    continue
                ref_id = match.group(1)
                if ref_id not in defined_ids:
                    broken.append(
                        f"step '{step.get('id')}' arg '{key}' references undefined step '${ref_id}'"
                    )

    if broken:
        logger.error("planner: broken placeholder references:\n  %s", "\n  ".join(broken))
        return [GET_USER_ID_STEP], False

    return all_steps, True


def _parse_planner_output(text: str) -> dict:
    clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
    data = json.loads(clean)
    route = data.get("route", "clarify")
    if route not in ("execute", "finalize", "clarify"):
        raise ValueError(f"unsupported route: {route}")

    if route == "execute":
        steps, is_valid = _validate_execute_steps(data.get("steps", []))
        if not is_valid:
            return {
                "route": "clarify",
                "reason": "invalid execute steps",
                "message": DEFAULT_CLARIFY_RESPONSE,
            }
        data["steps"] = steps

    if route in ("finalize", "clarify") and not str(data.get("message") or "").strip():
        data["message"] = DEFAULT_CLARIFY_RESPONSE if route == "clarify" else "Done."

    return data


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _extract_wallet_hint(text: str) -> str:
    match = _WALLET_HINT_RE.search(text)
    if not match:
        return ""
    hint = match.group(1).strip(" .,:;!?")
    return hint


def _normalize_intent_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _remove_amount_and_wallet(text: str) -> str:
    without_wallet = _WALLET_HINT_RE.sub("", text).strip()
    without_amount = _AMOUNT_TOKEN_RE.sub("", without_wallet, count=1).strip()
    return re.sub(r"\s+", " ", without_amount).strip(" .,:;!?")


def _is_generic_transaction_description(text: str) -> bool:
    normalized = _normalize_intent_text(text)
    if not normalized:
        return True
    return normalized in _GENERIC_TRANSACTION_DESCRIPTIONS


def _build_transaction_fallback(messages: list) -> dict | None:
    tools_schema = describe_tools()
    if "create_transaction" not in tools_schema:
        return None

    user_text = _latest_user_text(messages)
    if not user_text:
        return None

    amount_match = _AMOUNT_TOKEN_RE.search(user_text)
    if not amount_match:
        return None

    amount_token = amount_match.group(0).strip()
    if not amount_token:
        return None

    description = _remove_amount_and_wallet(user_text)
    if not description:
        return None

    args: dict[str, str] = {
        "amount": amount_token,
        "description": description,
    }

    return {
        "route": "execute",
        "reason": "planner clarify fallback for transaction entry",
        "steps": _ensure_get_user_id_first(
            [
                {
                    "id": "s1",
                    "type": "sql",
                    "reasoning": "create_transaction requires wallet_id, which is not yet known, so fetch candidate wallets for the authenticated user before writing the transaction.",
                    "description": "Fetch candidate wallets for the current user so wallet_id can be resolved later.",
                    "query_hint": "SELECT id, name, balance, currency FROM wallets WHERE user_id = $s0.user_id AND active = true",
                    "selection_mode": "none",
                },
                {
                    "id": "s2",
                    "type": "tool",
                    "name": "create_transaction",
                    "reasoning": "The request looks like a transaction entry with a concrete amount, so create the transaction after wallet candidates are available for resolution.",
                    "args": args,
                }
            ]
        ),
    }


async def planner_node(state: State) -> dict:
    messages = list(state.get("messages", []))
    history = [
        message for message in messages
        if isinstance(message, (HumanMessage, AIMessage))
        and not getattr(message, "tool_calls", None)
    ][-8:]

    llm = _get_llm()
    raw = ""
    try:
        response = await llm.ainvoke([SystemMessage(content=_build_planner_system_prompt())] + history)
        raw = response.content if hasattr(response, "content") else str(response)
        result = _parse_planner_output(raw)
    except json.JSONDecodeError as exc:
        logger.warning("planner: JSON parse failed: %s | raw=%s", exc, raw[:200])
        result = {
            "route": "clarify",
            "reason": "planner parse failure",
            "message": DEFAULT_CLARIFY_RESPONSE,
        }
    except ValueError as exc:
        logger.warning("planner: invalid planner output: %s | raw=%s", exc, raw[:200])
        result = {
            "route": "clarify",
            "reason": "planner validation failure",
            "message": DEFAULT_CLARIFY_RESPONSE,
        }
    except Exception as exc:
        logger.error("planner: unexpected failure: %s", exc)
        result = {
            "route": "clarify",
            "reason": "planner failure",
            "message": DEFAULT_CLARIFY_RESPONSE,
        }

    if result.get("route") == "clarify":
        fallback = _build_transaction_fallback(messages)
        if fallback is not None:
            logger.info("planner: replacing clarify with transaction execute fallback")
            result = fallback

    update: dict = {"route": result["route"]}
    if result["route"] == "execute":
        update["steps"] = result.get("steps", [])
        update["step_index"] = 0
        update["step_results"] = {}
        update["past_steps"] = list(state.get("past_steps") or [])
        update["replan_attempts"] = int(state.get("replan_attempts") or 0)
    else:
        update["messages"] = [AIMessage(content=str(result.get("message") or DEFAULT_CLARIFY_RESPONSE))]
    return update


def route_after_planner(state: State) -> Route:
    route = state.get("route", "clarify")
    if route not in ("execute", "finalize", "clarify"):
        logger.warning("route_after_planner: unexpected '%s', clarifying", route)
        return "clarify"
    return route


async def clarify_node(state: State) -> dict:
    existing_ai = next(
        (message for message in reversed(list(state.get("messages") or [])) if isinstance(message, AIMessage)),
        None,
    )
    if existing_ai is not None and str(getattr(existing_ai, "content", "")).strip():
        return {"messages": [existing_ai]}
    return {"messages": [AIMessage(content=DEFAULT_CLARIFY_RESPONSE)]}
