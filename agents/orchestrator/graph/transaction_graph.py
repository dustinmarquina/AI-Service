import json
from ..llm import get_extractor_llm
from langgraph.graph import StateGraph, END
from .state import State


async def extract_node(state: State):
    messages = state.get("messages", [])
    pending = state.get("pending_transaction") or {}

    if not messages:
        return {}

    user_input = messages[-1].content

    llm = get_extractor_llm()

    prompt = f"""
Extract transaction info from user input.

Current known data:
{pending}

User input:
"{user_input}"

Return ONLY JSON:
{{
  "amount": number or null,
  "description": string or null,
}}

Rules:
- If info already exists in current data, keep it
- If user adds new info, update it
"""

    response = await llm.ainvoke([
        {"role": "system", "content": "Return ONLY valid JSON"},
        {"role": "user", "content": prompt}
    ])

    try:
        extracted = json.loads(response.content)
    except:
        extracted = pending

    return {
        "pending_transaction": extracted
    }

async def validation_node(state: State):
    pending = state.get("pending_transaction")

    if not pending:
        return {}

    if pending.get("description") is None:
        return {
            "response": "Bạn đã chi cho việc gì vậy?"
        }

    if pending.get("amount") is None:
        return {
            "response": "Bạn đã chi bao nhiêu tiền?"
        }

    return {
        "tool_input": pending,
        "pending_transaction": None
    }


async def tool_node(state: State):
    tool_input = state.get("tool_input")

    if not tool_input:
        return {}

    payload = {
        "amount": tool_input["amount"],
        "description": tool_input["description"],
        "type": "EXPENSE"
    }

    # TODO: replace with real API call
    print("CALLING API WITH:", payload)

    return {
        "tool_output": payload,
        "response": "Đã lưu giao dịch thành công."
    }

def response_node(state: State):
    task = state.get("task_state")

    if task:
        missing = task.get("missing_slots", [])

        if missing:
            slot = missing[0]

            if task["intent"] == "sql":
                if slot == "condition":
                    return {
                        "response": "Bạn muốn lọc theo điều kiện nào?"
                    }

                if slot == "table":
                    return {
                        "response": "Bạn muốn truy vấn bảng nào?"
                    }

            if task["intent"] == "transaction":
                if slot == "amount":
                    return {
                        "response": "Bạn đã chi bao nhiêu?"
                    }

                if slot == "description":
                    return {
                        "response": "Bạn đã chi cho việc gì?"
                    }

    return {"response": "Tôi chưa hiểu, bạn có thể nói rõ hơn không?"}

def route_after_validation(state: State):
    if state.get("tool_input"):
        return "tool"
    return "response"




def build_transaction_graph():
    graph = StateGraph(State)

    graph.add_node("extract", extract_node)
    graph.add_node("validate", validation_node)
    graph.add_node("tool", tool_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("extract")

    graph.add_edge("extract", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "tool": "tool",
            "response": "response"
        }
    )

    graph.add_edge("tool", "response")
    graph.add_edge("response", END)

    return graph.compile()