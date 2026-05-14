import json
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.orchestrator.llm import get_classifier_llm
from .state import State

logger = logging.getLogger(__name__)

Route = Literal["sql_agent", "chat", "finalize", "clarify"]

PLANNER_SYSTEM_PROMPT = """You are a router for a Vietnamese personal finance assistant.

Given the user's latest message and conversation history, output ONLY a valid JSON object — no markdown, no explanation:

{"route": "<route>", "reason": "<one short sentence>"}

Routes:
- "sql_agent"  : user wants to READ, query, or summarize existing financial data
                 e.g. "tháng này tiêu bao nhiêu", "danh mục nào tốn nhất", "còn dư budget không"
- "chat"       : user wants to CREATE a transaction or category
                 e.g. "50k ăn sáng", "giải trí", "tạo category mới tên là du lịch"
- "finalize"   : user is chatting with no finance action needed
                 e.g. "cảm ơn", "ok", "bạn là ai"
- "clarify"    : the intent is genuinely ambiguous even with conversation history
                 e.g. a single word with no context, contradictory signals

Only use "clarify" when the route cannot be determined with reasonable confidence.
"""

_CLARIFY_RESPONSE = (
    "Mình chưa hiểu rõ ý bạn. "
    "Bạn muốn ghi một khoản chi tiêu, xem báo cáo, hay cần gì khác?"
)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_classifier_llm()
    return _llm


def _parse_route(text: str) -> tuple[Route, str]:
    """Extract route and reason from raw LLM output. Returns ('clarify', ...) on parse failure."""
    try:
        # Strip accidental markdown fences
        clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        data = json.loads(clean)
        route = data.get("route", "clarify")
        reason = data.get("reason", "")
        if route not in ("sql_agent", "chat", "finalize", "clarify"):
            logger.warning("planner: unknown route '%s', defaulting to clarify", route)
            return "clarify", reason
        return route, reason
    except (json.JSONDecodeError, AttributeError):
        logger.warning("planner: failed to parse output: %s", text[:200])
        return "clarify", "parse error"


def _fallback_route(messages: list[HumanMessage | AIMessage]) -> tuple[Route, str]:
    latest_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    text = (latest_human.content if latest_human else "") or ""
    normalized = " ".join(str(text).strip().lower().split())

    if not normalized:
        return "clarify", "empty message"

    query_markers = (
        "tổng",
        "bao nhiêu",
        "thống kê",
        "báo cáo",
        "liệt kê",
        "xem",
        "danh mục",
        "còn dư",
        "số dư",
        "vừa rồi",
        "tháng",
        "tuần",
        "hôm nay",
        "hôm qua",
    )
    create_markers = (
        "tạo category",
        "tao category",
        "tạo danh mục",
        "tao danh muc",
    )
    amount_pattern = r"\b\d+(?:[.,]\d+)?\s*(?:k|nghìn|ngàn|triệu|tr|đ|dong|vnd)\b"

    if any(marker in normalized for marker in create_markers):
        return "chat", "fallback matched category creation"

    if any(marker in normalized for marker in query_markers):
        return "sql_agent", "fallback matched reporting query"

    if re.search(amount_pattern, normalized):
        return "chat", "fallback matched transaction entry"

    if normalized in {"ok", "oke", "cảm ơn", "cam on", "hello", "hi"}:
        return "finalize", "fallback matched casual chat"

    return "clarify", "fallback could not determine intent"


async def planner_node(state: State) -> dict:
    """Classify intent and write the chosen route into state."""
    messages = list(state.get("messages", []))

    # Build a compact history (last 6 messages) + the current message for context
    history = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
    ][-6:]

    llm = _get_llm()
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + history
        )
    except Exception as exc:
        route, reason = _fallback_route(history)
        logger.warning("planner: classifier failed, using fallback route=%s reason=%s error=%s", route, reason, exc)
        return {"route": route}

    raw = response.content if hasattr(response, "content") else str(response)
    route, reason = _parse_route(raw)
    if route == "clarify" and reason == "parse error":
        route, reason = _fallback_route(history)

    logger.info("planner: route=%s reason=%s", route, reason)
    return {"route": route}


def route_after_planner(state: State) -> Route:
    """Conditional edge function — reads the route written by planner_node."""
    route = state.get("route", "clarify")
    if route not in ("sql_agent", "chat", "finalize", "clarify"):
        logger.warning("route_after_planner: unexpected route '%s', clarifying", route)
        return "clarify"
    return route


async def clarify_node(state: State) -> dict:
    """Emit a clarification message and let finalize save it to memory."""
    return {"messages": [AIMessage(content=_CLARIFY_RESPONSE)]}
