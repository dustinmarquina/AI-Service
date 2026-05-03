from .state import State
from langgraph.graph import StateGraph, END 
from ..llm import get_classifier_llm
from ..mcp_server import create_transaction


# --- Tool Registry (Phase 1: static mapping) ---
TOOL_REGISTRY = {
    "transaction": "create_transaction",
    # future:
    # "category": "create_category",
    # "sql": "query_sql"
}

TOOL_SCHEMA = [
    {
        "name": "create_transaction",
        "description": "record an expense transaction",
        "inputs": ["amount", "description"],
    }
]


# --- Simple Response Node ---
async def response_node(state: State):
    # If a response already exists (e.g., from MCP execution), keep it
    task = state.get("task_state") or {}

# ONLY reuse response if we just executed
    if task.get("next_action") != "ask_user" and "response" in state and state["response"]:
        return {"response": state["response"]}

    task = state.get("task_state") or {}

    missing = task.get("missing_slots", [])

    if missing:
        llm = get_classifier_llm()

        prompt = f"""
You are a helpful assistant asking the user for missing information.

Current task:
{task}

Missing fields:
{missing}

Instructions:
- Ask a natural, short question to get the missing information
- Use Vietnamese
- Do NOT mention technical terms like 'slot' or 'field'
- If only one missing value → ask directly
- If multiple → ask the most important one first

Return ONLY the question.
"""

        response = await llm.ainvoke([
            {"role": "system", "content": "You ask for missing information."},
            {"role": "user", "content": prompt}
        ])

        return {"response": response.content.strip()}

    return {"response": "OK"}


# --- Task Extraction Node ---
async def extract_task_node(state: State):
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)
    existing_task = state.get("task_state")

    llm = get_classifier_llm()
    import json

    if existing_task:
        # Deterministic continuation
        if existing_task.get("status") == "collecting":
            decision_text = "continue"
        else:
            decision_text = "new_task"

        # -----------------------------
        # STEP 2: CONTINUE FLOW
        # -----------------------------
        if decision_text == "continue":
            required = existing_task.get("required_slots", [])
            filled = existing_task.get("filled_slots", {})

            missing = [
                k for k in required
                if k not in filled or filled[k] is None or str(filled[k]).strip() == ""
            ]

            # --- Deterministic slot fill (critical fix) ---
            if len(missing) == 1:
                slot = missing[0]
                text = user_input.strip().lower()

                def is_amount_like(t):
                    return t.endswith("k") or t.replace(",", "").isdigit()

                if slot == "description":
                    # prevent assigning amount as description
                    if is_amount_like(text):
                        return {"task_state": existing_task}
                    filled[slot] = text

                elif slot == "amount":
                    # only accept valid amount
                    if not is_amount_like(text):
                        return {"task_state": existing_task}
                    filled[slot] = text

                existing_task["filled_slots"] = filled
                existing_task["missing_slots"] = []
                existing_task["next_action"] = "execute"
                existing_task["status"] = "ready"

                return {"task_state": existing_task}

            # fallback to LLM if multiple slots
            fill_prompt = f"""
You are filling slots for a task.

Current task:
{existing_task}

User message:
"{user_input}"

Rules:
- Prefer filling missing slots over changing intent
- If user provides partial info, extract it
- Do NOT change task intent

Return ONLY JSON of filled_slots.
"""

            fill_response = await llm.ainvoke([
                {"role": "system", "content": "Extract slot values."},
                {"role": "user", "content": fill_prompt}
            ])

            try:
                new_filled = json.loads(fill_response.content)
            except:
                new_filled = {}

            required = existing_task.get("required_slots", [])
            filtered_new = {k: v for k, v in new_filled.items() if k in required}

            filled = existing_task.get("filled_slots", {})
            filled.update(filtered_new)

            missing = [
                k for k in required
                if k not in filled or filled[k] is None or str(filled[k]).strip() == ""
            ]

            existing_task["filled_slots"] = filled
            existing_task["missing_slots"] = missing

            if not missing:
                existing_task["next_action"] = "execute"
                existing_task["status"] = "ready"
            else:
                existing_task["next_action"] = "ask_user"
                existing_task["status"] = "collecting"

            return {"task_state": existing_task}

    # -----------------------------
    # STEP 3: TOOL-FIRST EXTRACTION
    # -----------------------------
    system_prompt = """
You are a tool selector and argument extractor.

Available tool:
create_transaction:
- purpose: record a financial expense
- inputs:
  - amount (e.g. "50k")
  - description (e.g. "ăn sáng", "cà phê")

Rules:
- If the message implies spending (food, drink, transport, buying, activities), choose create_transaction
- Extract as many arguments as possible
- Partial arguments are allowed
- If no tool applies, return tool = "unknown"

Examples:
"50k ăn sáng" → {"tool": "create_transaction", "arguments": {"amount": "50k", "description": "ăn sáng"}}
"bún bò huế" → {"tool": "create_transaction", "arguments": {"description": "bún bò huế"}}

Return JSON:
{
  "tool": "...",
  "arguments": {...}
}
"""

    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ])

    try:
        parsed = json.loads(response.content)
    except:
        parsed = {"tool": "unknown", "arguments": {}}

    tool = parsed.get("tool")
    arguments = parsed.get("arguments", {})

    # fallback: keep existing task if unknown
    if existing_task and tool == "unknown":
        return {"task_state": existing_task}

    # derive slots from tool schema
    TOOL_SCHEMAS = {
        "create_transaction": ["amount", "description"]
    }

    if tool not in TOOL_SCHEMAS:
        return {
            "task_state": {
                "tool": None,
                "required_slots": [],
                "filled_slots": {},
                "missing_slots": [],
                "next_action": "ask_user"
            }
        }

    required = TOOL_SCHEMAS[tool]
    filled = {k: v for k, v in arguments.items() if k in required}

    missing = [
        k for k in required
        if k not in filled or filled[k] is None or str(filled[k]).strip() == ""
    ]

    task_state = {
        "tool": tool,
        "required_slots": required,
        "filled_slots": filled,
        "missing_slots": missing,
        "next_action": "execute" if not missing else "ask_user",
        "status": "ready" if not missing else "collecting"
    }

    return {"task_state": task_state}



# --- MCP Execution Node ---
async def execute_mcp_node(state: State):
    task = state.get("task_state") or {}
    # intent = task.get("intent")
    slots = task.get("filled_slots", {})
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZjFlYjIwYi01NGU1LTQ5MTAtYjk3NC02MDE1MTE4NWU0OGYiLCJhY2NvdW50SWQiOiIzNTcwMmZiMy04NjYyLTRmYmMtOGZiNC00Mjc3YTUwNGM3YmEiLCJlbWFpbCI6ImRlbW8wMDFAZ21haWwuY29tIiwicm9sZXMiOiJST0xFX1VTRVIiLCJpYXQiOjE3Nzc3Mjg5OTksImV4cCI6MTc3NzgxNTM5OX0.aKWSZNPJSSt0V5Rli03RxLxJFe1d9qQCK-t0nLx9scw"
    try:
        tool_name = task.get("tool")

        if not tool_name:
            return {"response": "Không tìm thấy tool phù hợp"}

        if tool_name == "create_transaction":
            raw_amount = str(slots.get("amount", ""))

            # normalize amount (e.g. "20k" -> 20000)
            def normalize_amount(val: str):
                val = val.lower().replace(",", "").strip()
                if val.endswith("k"):
                    try:
                        return int(val[:-1]) * 1000
                    except:
                        return None
                if val.isdigit():
                    return int(val)
                return None

            normalized_amount = normalize_amount(raw_amount)

            if normalized_amount is None:
                return {"response": "Số tiền không hợp lệ"}

            tool_result = await create_transaction(
                amount=normalized_amount,
                description=str(slots.get("description", "")),
                token=token,
            )

            if tool_result.get("status") == "error":
                error_data = tool_result.get("data")
                if error_data is not None:
                    return {"response": f"Lỗi backend: {error_data}"}
                return {"response": f"Lỗi backend: {tool_result.get('message', 'Unknown error')}"}

            data = tool_result.get("data") or {}
            return {
                "response": f"Đã ghi giao dịch: {data.get('amount')} - {data.get('description')}"
                # reset task state after execution
                
            }

    except Exception as e:
        return {
            "response": f"Lỗi khi gọi backend: {str(e)}"
        }

    return {"response": "Không xác định được hành động"}


# --- Routing by Task ---
def route_by_task(state: State):
    task = state.get("task_state") or {}
    next_action = task.get("next_action")

    if next_action == "execute":
        return "execute"

    return "response"

def build_main_graph():
    graph = StateGraph(State)

    # --- nodes ---
    graph.add_node("extract_task", extract_task_node)
    graph.add_node("execute", execute_mcp_node)
    graph.add_node("response", response_node)

    # --- entry ---
    graph.set_entry_point("extract_task")

    # --- routing ---
    graph.add_conditional_edges(
        "extract_task",
        route_by_task,
        {
            "execute": "execute",
            "response": "response",
        }
    )

    graph.add_edge("execute", "response")
    graph.add_edge("response", END)

    return graph.compile()