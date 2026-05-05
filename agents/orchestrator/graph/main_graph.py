import os
import re
import shlex
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
import logging

from .state import State


load_dotenv()


MAX_MEMORY_MESSAGES = 12
SHORT_TERM_MEMORY: dict[str, InMemoryChatMessageHistory] = {}
MCP_CLIENTS: list[Any] = []



def _get_session_id(state: State) -> str:
    return str(state.get("session_id") or state.get("user_id") or "default")


def _get_memory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SHORT_TERM_MEMORY:
        SHORT_TERM_MEMORY[session_id] = InMemoryChatMessageHistory()
    return SHORT_TERM_MEMORY[session_id]


def _trim_memory(history: InMemoryChatMessageHistory) -> None:
    if len(history.messages) > MAX_MEMORY_MESSAGES:
        history.messages = history.messages[-MAX_MEMORY_MESSAGES:]



def _build_mcp_server_config() -> dict[str, dict[str, Any]]:
    command = os.getenv("ORCHESTRATOR_MCP_COMMAND", "python")
    args = os.getenv("ORCHESTRATOR_MCP_ARGS", "-m agents.orchestrator.mcp_server")

    return {
        "orchestrator": {
            "transport": os.getenv("ORCHESTRATOR_MCP_TRANSPORT", "stdio"),
            "command": command,
            "args": shlex.split(args),
        }
    }


def _field_names_from_tool(tool: Any) -> set[str]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return set()

    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())

    schema_fields = getattr(schema, "__fields__", None)
    if isinstance(schema_fields, dict):
        return set(schema_fields.keys())

    return set()


async def build_main_graph():
    client = MultiServerMCPClient(_build_mcp_server_config())
    MCP_CLIENTS.append(client)

    tools = await client.get_tools()
    tool_node = ToolNode(tools)
    tool_arg_fields = {
        getattr(tool, "name", ""): _field_names_from_tool(tool)
        for tool in tools
    }

    tools_catalog = "\n".join(
        f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '')}".strip()
        for tool in tools
    ) or "- No MCP tools available"

    llm = ChatGroq(
        model=os.getenv("ORCHESTRATOR_LLM_MODEL", "openai/gpt-oss-120b"),
        temperature=0.2,
        api_key=os.getenv("groq_api_key"),
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a Vietnamese finance assistant with short-term memory.

You have access to MCP tools discovered at runtime:
{tools_catalog}

Rules:
- Use the conversation history to resolve references like 'that one', 'same as before', or omitted details.
- Do not do explicit intent detection or slot filling.
- Call the appropriate MCP tool whenever the user wants a structured finance action.
- If the user is just chatting or the request is incomplete, respond naturally in Vietnamese.
- Keep replies short and clear.
""",
        ),
        MessagesPlaceholder("messages"),
    ])

    llm_with_tools = llm.bind_tools(tools)
    chat_chain = prompt | llm_with_tools

    async def chat_node(state: State):
        session_id = _get_session_id(state)
        history = _get_memory(session_id)
        current_messages = list(state.get("messages", []))
        prompt_messages = history.messages + current_messages

        response = await chat_chain.ainvoke({"messages": prompt_messages})
        return {"messages": [response]}

    def inject_runtime_context_node(state: State):
        current_messages = list(state.get("messages", []))
        if not current_messages:
            return {}

        last_ai = next(
            (message for message in reversed(current_messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not getattr(last_ai, "tool_calls", None):
            return {}
        runtime_token = str(state.get("token") or os.getenv("TRANSACTION_API_TOKEN", "")).strip()
        runtime_user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()
        print(f"at main_graph runtime_user_id: {runtime_user_id}")

        logger = logging.getLogger(__name__)
        redacted = (runtime_token[:8] + "...") if runtime_token else "(none)"
        logger.info("inject_runtime_context: runtime_token=%s user_id=%s", redacted, runtime_user_id or "(none)")

        # Log incoming tool_calls for debugging
        try:
            for idx, call in enumerate(last_ai.tool_calls):
                name = call.get("name")
                args = call.get("args") or {}
                logger.info("inject_runtime_context: found tool_call[%d] name=%s args_keys=%s", idx, name, list(args.keys()))
        except Exception:
            logger.exception("inject_runtime_context: failed to log incoming tool_calls")

        updated_tool_calls = []
        changed = False
        injected_summary: list[dict[str, object]] = []

        for call in last_ai.tool_calls:
            call_copy = dict(call)
            args = dict(call_copy.get("args") or {})
            tool_name = str(call_copy.get("name", ""))
            fields = tool_arg_fields.get(tool_name, set())

            if runtime_token and not args.get("token"):
                args["token"] = runtime_token
                changed = True
                injected_summary.append({"tool": tool_name, "injected": "token"})
                logger.info("inject_runtime_context: injected 'token' into tool '%s'", tool_name)

            if runtime_user_id:
                # Inject user id in the form expected by the tool schema.
                if "userId" in fields:
                    if not args.get("userId"):
                        args["userId"] = runtime_user_id
                        changed = True
                        injected_summary.append({"tool": tool_name, "injected": "userId"})
                        logger.info("inject_runtime_context: injected 'userId' into tool '%s'", tool_name)
                else:
                    # default to snake_case 'user_id' which is commonly used by our tools
                    if not args.get("user_id"):
                        args["user_id"] = runtime_user_id
                        changed = True
                        injected_summary.append({"tool": tool_name, "injected": "user_id"})
                        logger.info("inject_runtime_context: injected 'user_id' into tool '%s'", tool_name)

            # print(f"Processing tool call: {call_copy.get('name')} with args {args}")    
            call_copy["args"] = args
            updated_tool_calls.append(call_copy)
            print(f"Updated tool call: {call_copy.get('name')} with args {args}")

        if not changed:
            return {}

        updated_ai = AIMessage(
            content=last_ai.content,
            additional_kwargs=last_ai.additional_kwargs,
            response_metadata=last_ai.response_metadata,
            id=last_ai.id,
            tool_calls=updated_tool_calls,
            invalid_tool_calls=getattr(last_ai, "invalid_tool_calls", []),
            name=last_ai.name,
        )

        return {"messages": [updated_ai], "injected": injected_summary}

    async def finalize_node(state: State):
        session_id = _get_session_id(state)
        history = _get_memory(session_id)
        current_messages = list(state.get("messages", []))

        last_user_message = next(
            (message for message in reversed(current_messages) if isinstance(message, HumanMessage)),
            None,
        )
        last_ai_message = next(
            (
                message
                for message in reversed(current_messages)
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
            ),
            None,
        )

        if last_user_message:
            history.add_messages([last_user_message])

        if last_ai_message:
            history.add_messages([last_ai_message])
            _trim_memory(history)
            return {"response": last_ai_message}

        fallback = AIMessage(content="Mình chưa có phản hồi phù hợp.")
        history.add_messages([fallback])
        _trim_memory(history)
        return {"response": fallback}

    graph = StateGraph(State)

    graph.add_node("chat", chat_node)
    graph.add_node("inject_runtime_context", inject_runtime_context_node)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges(
        "chat",
        tools_condition,
        {
            "tools": "inject_runtime_context",
            "__end__": "finalize",
        },
    )
    graph.add_edge("inject_runtime_context", "tools")
    graph.add_edge("tools", "chat")
    graph.add_edge("finalize", END)

    return graph.compile()
#     return [
#         k for k in required
#         if k not in filled or filled[k] is None or str(filled[k]).strip() == ""
#     ]


# def parse_json_safe(text: str, fallback: dict) -> dict:
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         return fallback


# async def invoke_llm(llm, system: str, user: str) -> str:
#     response = await llm.ainvoke([
#         SystemMessage(content=system),
#         HumanMessage(content=user)
#     ])
#     return response.content.strip()


# def build_task(intent: str, arguments: dict) -> dict:
#     """Construct a fresh task state from a tool intent and extracted arguments."""
#     required = TOOL_REGISTRY[intent]["required_slots"]
#     filled = {k: v for k, v in arguments.items() if k in required}
#     missing = compute_missing_slots(required, filled)
#     return {
#         "tool": intent,
#         "required_slots": required,
#         "filled_slots": filled,
#         "missing_slots": missing,
#         "next_action": "execute" if not missing else "ask_user",
#         "status": "ready" if not missing else "collecting"
#     }


# # ------------------------------------
# # Intent Classifier
# # ------------------------------------

# async def classify_intent(llm, user_input: str, active_task: dict | None) -> dict:
#     meta_list = "\n".join([
#         f"- {name}: {meta['description']}"
#         for name, meta in META_INTENTS.items()
#     ])

#     tool_list = "\n".join([
#         f"- {name}: {meta['description']} | required slots: {meta['required_slots']}"
#         for name, meta in TOOL_REGISTRY.items()
#     ])

#     active_context = ""
#     if active_task:
#         active_context = f"""
# Currently active task: {active_task.get('tool')}
# Missing slots: {active_task.get('missing_slots', [])}
# """

#     system_prompt = f"""
# You are an intent classifier for a Vietnamese personal finance chatbot.

# Meta intents (always available):
# {meta_list}

# Tool intents (action-based):
# {tool_list}

# {active_context}

# Rules:
# - Choose the single best matching intent
# - For tool intents, also extract any slot values present in the message
# - Use confidence "low" when the message is ambiguous or could match multiple intents
# - If the active task is collecting and the message provides a value → prefer the active tool intent
# - Never return an intent not listed above

# Return ONLY JSON:
# {{
#   "intent": "...",
#   "confidence": "high" | "low",
#   "arguments": {{...}}
# }}
# """

#     raw = await invoke_llm(llm, system_prompt, user_input)
#     result = parse_json_safe(raw, fallback={
#         "intent": "out_of_scope",
#         "confidence": "low",
#         "arguments": {}
#     })

#     result.setdefault("intent", "out_of_scope")
#     result.setdefault("confidence", "low")
#     result.setdefault("arguments", {})

#     all_intents = set(META_INTENTS.keys()) | set(TOOL_REGISTRY.keys())
#     if result["intent"] not in all_intents:
#         result["intent"] = "out_of_scope"
#         result["confidence"] = "low"

#     return result


# # ------------------------------------
# # Response Node
# # ------------------------------------

# async def response_node(state: State):
#     next_action = state.get("next_action")
#     task = state.get("task_state") or {}
#     task_stack = state.get("task_stack", [])
#     llm = get_classifier_llm()

#     if next_action == "chitchat":
#         user_input = state.get("user_input", "")
#         prompt = f"""
# You are a friendly Vietnamese personal finance assistant.
# The user said something casual or off-topic. Respond naturally and briefly in Vietnamese.
# If there is an active task, gently remind them after your reply.

# Active task: {json.dumps(task, ensure_ascii=False) if task else "none"}
# User message: "{user_input}"
# """
#         answer = await invoke_llm(llm, "Respond naturally in Vietnamese.", prompt)
#         return {"response": answer}

#     if next_action == "out_of_scope":
#         return {"response": "Mình chưa hỗ trợ yêu cầu này. Bạn có muốn ghi một giao dịch không?"}

#     if next_action == "clarify" and state.get("clarify_reason") == "low_confidence":
#         return {"response": "Bạn có thể nói rõ hơn không? Mình chưa hiểu ý bạn."}

#     if next_action == "clarify" and state.get("clarify_reason") == "user_asked":
#         if task:
#             missing = task.get("missing_slots", [])
#             prompt = f"""
# You are a helpful Vietnamese assistant.
# The user wants to understand what you are currently asking them.

# Current task: {json.dumps(task, indent=2, ensure_ascii=False)}
# Missing slots: {missing}

# Explain what you are trying to do and what information you still need.
# Use plain Vietnamese. Do NOT use technical terms.
# """
#             answer = await invoke_llm(llm, "Explain the current task.", prompt)
#             return {"response": answer}
#         return {"response": "Mình chưa có yêu cầu nào đang thực hiện. Bạn muốn làm gì?"}

#     if next_action == "confirm_cancel":
#         return {"response": "Bạn có chắc muốn hủy không? (có / không)"}

#     if next_action == "cancelled":
#         # If there's a paused task on the stack, offer to resume
#         if task_stack:
#             paused = task_stack[-1]
#             return {
#                 "response": f"Đã hủy. Trước đó bạn đang ghi {paused.get('tool')} — bạn có muốn tiếp tục không?",
#                 "task_state": paused,
#                 "task_stack": task_stack[:-1]
#             }
#         return {"response": "Đã hủy. Bạn muốn làm gì tiếp theo?"}

#     if next_action == "switched":
#         new_tool = task.get("tool", "")
#         paused_count = len(task_stack)
#         resume_hint = f" Mình sẽ nhớ task trước cho bạn." if paused_count > 0 else ""
#         return {"response": f"Được rồi, mình sẽ xử lý {new_tool} trước nhé.{resume_hint}"}

#     if task.get("next_action") != "ask_user" and "response" in state and state["response"]:
#         return {"response": state["response"]}

#     missing = task.get("missing_slots", [])

#     if missing:
#         prompt = f"""
# You are a helpful assistant asking the user for missing information.

# Current task:
# {json.dumps(task, indent=2, ensure_ascii=False)}

# Missing fields:
# {missing}

# Instructions:
# - Ask a natural, short question to get the missing information
# - Use Vietnamese
# - Do NOT mention technical terms like 'slot' or 'field'
# - If only one missing value → ask directly
# - If multiple → ask the most important one first

# Return ONLY the question.
# """
#         answer = await invoke_llm(llm, "You ask for missing information.", prompt)
#         return {"response": answer}

#     return {"response": "OK"}


# # ------------------------------------
# # Extract Task Node
# # ------------------------------------

# async def extract_task_node(state: State):
#     messages = state.get("messages", [])
#     if not messages:
#         return {}

#     last_message = messages[-1]
#     user_input = last_message.content if hasattr(last_message, "content") else str(last_message)
#     existing_task = state.get("task_state")
#     task_stack = state.get("task_stack", [])

#     llm = get_classifier_llm()

#     # ── SINGLE CLASSIFICATION CALL ──────────────────────────────────
#     classification = await classify_intent(llm, user_input, existing_task)
#     intent = classification["intent"]
#     confidence = classification["confidence"]
#     arguments = classification["arguments"]

#     # ── LOW CONFIDENCE ──────────────────────────────────────────────
#     if confidence == "low":
#         return {
#             "task_state": existing_task,
#             "task_stack": task_stack,
#             "next_action": "clarify",
#             "clarify_reason": "low_confidence",
#             "user_input": user_input,
#         }

#     # ── META-INTENT BRANCHING ───────────────────────────────────────
#     if intent == "chitchat":
#         return {
#             "task_state": existing_task,
#             "task_stack": task_stack,
#             "next_action": "chitchat",
#             "user_input": user_input,
#         }

#     if intent == "out_of_scope":
#         return {
#             "task_state": existing_task,
#             "task_stack": task_stack,
#             "next_action": "out_of_scope",
#             "user_input": user_input,
#         }

#     if intent == "clarify":
#         return {
#             "task_state": existing_task,
#             "task_stack": task_stack,
#             "next_action": "clarify",
#             "clarify_reason": "user_asked",
#             "user_input": user_input,
#         }

#     if intent == "cancel":
#         if existing_task and existing_task.get("status") == "collecting":
#             return {
#                 "task_state": existing_task,
#                 "task_stack": task_stack,
#                 "next_action": "confirm_cancel",
#                 "user_input": user_input,
#             }
#         return {
#             "task_state": None,
#             "task_stack": task_stack,
#             "next_action": "chitchat",
#             "user_input": user_input,
#         }

#     # ── TOOL INTENT ─────────────────────────────────────────────────
#     if intent not in TOOL_REGISTRY:
#         return {
#             "task_state": existing_task,
#             "task_stack": task_stack,
#             "next_action": "out_of_scope",
#             "user_input": user_input,
#         }

#     # ── SWITCH DETECTION ────────────────────────────────────────────  ← NEW
#     if existing_task and existing_task.get("status") == "collecting":
#         active_tool = existing_task.get("tool")

#         if intent != active_tool:
#             # Different tool detected — pause current, start new
#             new_task = build_task(intent, arguments)
#             return {
#                 "task_state": new_task,
#                 "task_stack": task_stack + [existing_task],  # push to stack
#                 "next_action": "switched",
#                 "user_input": user_input,
#             }

#         # ── CONTINUATION ─────────────────────────────────────────────
#         required = existing_task.get("required_slots", [])
#         filled = existing_task.get("filled_slots", {}).copy()
#         missing = compute_missing_slots(required, filled)

#         if len(missing) == 1:
#             slot = missing[0]

#             # Use classifier arguments — no heuristics needed       ← is_amount_like REMOVED
#             if slot in arguments and arguments[slot]:
#                 filled[slot] = arguments[slot]
#             else:
#                 # Classifier returned nothing for this slot —
#                 # treat entire user input as the slot value
#                 filled[slot] = user_input.strip()

#             missing = compute_missing_slots(required, filled)
#             existing_task.update({
#                 "filled_slots": filled,
#                 "missing_slots": missing,
#                 "next_action": "execute" if not missing else "ask_user",
#                 "status": "ready" if not missing else "collecting"
#             })
#             return {
#                 "task_state": existing_task,
#                 "task_stack": task_stack,
#             }

#         # Multi-slot continuation
#         filtered = {k: v for k, v in arguments.items() if k in required}

#         if filtered:
#             filled.update(filtered)
#         else:
#             fill_prompt = f"""
# You are filling slots for a task.

# Current task:
# {json.dumps(existing_task, indent=2, ensure_ascii=False)}

# User message:
# "{user_input}"

# Rules:
# - Only fill slots that belong to the current task
# - If user provides partial info, extract what you can
# - Do NOT change task intent

# Return ONLY a flat JSON object of slot key-value pairs.
# """
#             raw = await invoke_llm(llm, "Extract slot values.", fill_prompt)
#             new_filled = parse_json_safe(raw, fallback={})
#             filtered = {k: v for k, v in new_filled.items() if k in required}
#             filled.update(filtered)

#         missing = compute_missing_slots(required, filled)
#         existing_task.update({
#             "filled_slots": filled,
#             "missing_slots": missing,
#             "next_action": "execute" if not missing else "ask_user",
#             "status": "ready" if not missing else "collecting"
#         })
#         return {
#             "task_state": existing_task,
#             "task_stack": task_stack,
#         }

#     # ── NEW TASK ─────────────────────────────────────────────────────
#     new_task = build_task(intent, arguments)
#     return {
#         "task_state": new_task,
#         "task_stack": task_stack,
#         "user_input": user_input,
#     }


# # ------------------------------------
# # Router
# # ------------------------------------

# def route_by_task(state: State):
#     next_action = state.get("next_action")

#     if next_action in ("chitchat", "out_of_scope", "clarify", "confirm_cancel", "cancelled", "switched"):
#         return "response"

#     task = state.get("task_state") or {}
#     if task.get("next_action") == "execute":
#         return "execute"

#     return "response"


# # ------------------------------------
# # Updated State
# # ------------------------------------

# # Add task_stack to your State TypedDict:
# #
# # class State(TypedDict):
# #     messages: list
# #     task_state: dict | None
# #     task_stack: list           ← NEW
# #     next_action: str | None
# #     clarify_reason: str | None
# #     user_input: str | None
# #     response: str | None
# #     token: str | None



# # --- MCP Execution Node ---
# async def execute_mcp_node(state: State):
#     task = state.get("task_state") or {}
#     # intent = task.get("intent")
#     slots = task.get("filled_slots", {})
#     token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZjFlYjIwYi01NGU1LTQ5MTAtYjk3NC02MDE1MTE4NWU0OGYiLCJhY2NvdW50SWQiOiIzNTcwMmZiMy04NjYyLTRmYmMtOGZiNC00Mjc3YTUwNGM3YmEiLCJlbWFpbCI6ImRlbW8wMDFAZ21haWwuY29tIiwicm9sZXMiOiJST0xFX1VTRVIiLCJpYXQiOjE3Nzc4MTU4ODcsImV4cCI6MTc3NzkwMjI4N30.egLG2RbhT5YgtOe0qVbll-4SqMyIchdHX2tkXW5wNW8"
#     try:
#         tool_name = task.get("tool")

#         if not tool_name:
#             return {"response": "Không tìm thấy tool phù hợp"}

#         if tool_name == "create_transaction":
#             raw_amount = str(slots.get("amount", ""))

#             # normalize amount (e.g. "20k" -> 20000)
#             def normalize_amount(val: str):
#                 val = val.lower().replace(",", "").strip()
#                 if val.endswith("k"):
#                     try:
#                         return int(val[:-1]) * 1000
#                     except:
#                         return None
#                 if val.isdigit():
#                     return int(val)
#                 return None

#             normalized_amount = normalize_amount(raw_amount)

#             if normalized_amount is None:
#                 return {"response": "Số tiền không hợp lệ"}

#             tool_result = await create_transaction(
#                 amount=normalized_amount,
#                 description=str(slots.get("description", "")),
#                 token=token,
#             )

#             if tool_result.get("status") == "error":
#                 error_data = tool_result.get("data")
#                 if error_data is not None:
#                     return {"response": f"Lỗi backend: {error_data}"}
#                 return {"response": f"Lỗi backend: {tool_result.get('message', 'Unknown error')}"}

#             data = tool_result.get("data") or {}
#             return {
#                 "response": f"Đã ghi giao dịch: {data.get('amount')} - {data.get('description')}"
#             }

#     except Exception as e:
#         return {
#             "response": f"Lỗi khi gọi backend: {str(e)}"
#         }

#     return {"response": "Không xác định được hành động"}


# # --- Routing by Task ---
# def route_by_task(state: State):
#     task = state.get("task_state") or {}
#     next_action = task.get("next_action")

#     if next_action == "execute":
#         return "execute"

#     return "response"

# def build_main_graph():
#     graph = StateGraph(State)

#     # --- nodes ---
#     graph.add_node("extract_task", extract_task_node)
#     graph.add_node("execute", execute_mcp_node)
#     graph.add_node("response", response_node)

#     # --- entry ---
#     graph.set_entry_point("extract_task")

#     # --- routing ---
#     graph.add_conditional_edges(
#         "extract_task",
#         route_by_task,
#         {
#             "execute": "execute",
#             "response": "response",
#         }
#     )

#     graph.add_edge("execute", "response")
#     graph.add_edge("response", END)

#     return graph.compile()
# import json
# import os
# import shlex
# from typing import Any, Dict

# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.graph import END, START, StateGraph
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_mcp_adapters.client import MultiServerMCPClient
# load_dotenv()
# from .state import State


# MAX_MEMORY_MESSAGES = 12
# SHORT_TERM_MEMORY: dict[str, InMemoryChatMessageHistory] = {}
# MCP_CLIENTS: list[Any] = []


# def _get_session_id(state: State) -> str:
#     return str(state.get("session_id") or state.get("user_id") or "default")


# def _get_memory(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in SHORT_TERM_MEMORY:
#         SHORT_TERM_MEMORY[session_id] = InMemoryChatMessageHistory()
#     return SHORT_TERM_MEMORY[session_id]


# def _trim_memory(history: InMemoryChatMessageHistory) -> None:
#     if len(history.messages) > MAX_MEMORY_MESSAGES:
#         history.messages = history.messages[-MAX_MEMORY_MESSAGES:]


# def _message_text(message: BaseMessage) -> str:
#     return message.content if hasattr(message, "content") else str(message)


# def _recent_history_text(history: InMemoryChatMessageHistory) -> str:
#     recent_messages = history.messages[-MAX_MEMORY_MESSAGES:]
#     if not recent_messages:
#         return "(empty)"

#     lines = []
#     for message in recent_messages:
#         role = getattr(message, "type", "message")
#         lines.append(f"{role}: {_message_text(message)}")
#     return "\n".join(lines)


# def _build_mcp_server_config() -> dict[str, dict[str, Any]]:
#     command = os.getenv("ORCHESTRATOR_MCP_COMMAND", "python")
#     args = os.getenv("ORCHESTRATOR_MCP_ARGS", "-m agents.orchestrator.mcp_server")

#     return {
#         "orchestrator": {
#             "transport": os.getenv("ORCHESTRATOR_MCP_TRANSPORT", "stdio"),
#             "command": command,
#             "args": shlex.split(args),
#         }
#     }


# def _field_names_from_tool(tool: Any) -> set[str]:
#     schema = getattr(tool, "args_schema", None)
#     if schema is None:
#         return set()

#     fields = getattr(schema, "model_fields", None)
#     if isinstance(fields, dict):
#         return set(fields.keys())

#     schema_fields = getattr(schema, "__fields__", None)
#     if isinstance(schema_fields, dict):
#         return set(schema_fields.keys())

#     return set()


# async def build_main_graph():
#     client = MultiServerMCPClient(_build_mcp_server_config())
#     MCP_CLIENTS.append(client)

#     tools = await client.get_tools()
#     tool_node = ToolNode(tools)
#     tool_arg_fields = {
#         getattr(tool, "name", ""): _field_names_from_tool(tool)
#         for tool in tools
#     }

#     tools_catalog = "\n".join(
#         f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '')}".strip()
#         for tool in tools
#     ) or "- No MCP tools available"

#     llm = ChatGroq(
#         model=os.getenv("ORCHESTRATOR_LLM_MODEL", "openai/gpt-oss-120b"),
#         temperature=0.2,
#         api_key=os.getenv("groq_api_key"),
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             f"""You are a Vietnamese finance assistant with short-term memory.

# You have access to MCP tools discovered at runtime:
# {tools_catalog}

# Rules:
# - Use the conversation history to resolve references like 'that one', 'same as before', or omitted details.
# - Do not do explicit intent detection or slot filling.
# - Call the appropriate MCP tool whenever the user wants a structured finance action.
# - If the user is just chatting or the request is incomplete, respond naturally in Vietnamese.
# - Keep replies short and clear.
# """,
#         ),
#         MessagesPlaceholder("messages"),
#     ])

#     llm_with_tools = llm.bind_tools(tools)
#     chat_chain = prompt | llm_with_tools

#     async def chat_node(state: State):
#         session_id = _get_session_id(state)
#         history = _get_memory(session_id)
#         current_messages = list(state.get("messages", []))
#         prompt_messages = history.messages + current_messages

#         response = await chat_chain.ainvoke({"messages": prompt_messages})
#         return {"messages": [response]}

#     def inject_runtime_context_node(state: State):
#         current_messages = list(state.get("messages", []))
#         if not current_messages:
#             return {}

#         last_ai = next(
#             (message for message in reversed(current_messages) if isinstance(message, AIMessage)),
#             None,
#         )
#         if last_ai is None or not getattr(last_ai, "tool_calls", None):
#             return {}

#         runtime_token = str(state.get("token") or os.getenv("TRANSACTION_API_TOKEN", "")).strip()
#         runtime_user_id = str(state.get("user_id") or os.getenv("TRANSACTION_USER_ID", "")).strip()

#         updated_tool_calls = []
#         changed = False

#         for call in last_ai.tool_calls:
#             call_copy = dict(call)
#             args = dict(call_copy.get("args") or {})
#             tool_name = str(call_copy.get("name", ""))
#             fields = tool_arg_fields.get(tool_name, set())

#             if runtime_token and "token" in fields and not args.get("token"):
#                 args["token"] = runtime_token
#                 changed = True

#             if runtime_user_id:
#                 if "user_id" in fields and not args.get("user_id"):
#                     args["user_id"] = runtime_user_id
#                     changed = True
#                 if "userId" in fields and not args.get("userId"):
#                     args["userId"] = runtime_user_id
#                     changed = True

#             call_copy["args"] = args
#             updated_tool_calls.append(call_copy)

#         if not changed:
#             return {}

#         updated_ai = AIMessage(
#             content=last_ai.content,
#             additional_kwargs=last_ai.additional_kwargs,
#             response_metadata=last_ai.response_metadata,
#             id=last_ai.id,
#             tool_calls=updated_tool_calls,
#             invalid_tool_calls=getattr(last_ai, "invalid_tool_calls", []),
#             name=last_ai.name,
#         )

#         return {"messages": [updated_ai]}

#     async def finalize_node(state: State):
#         session_id = _get_session_id(state)
#         history = _get_memory(session_id)
#         current_messages = list(state.get("messages", []))

#         last_user_message = next(
#             (message for message in reversed(current_messages) if isinstance(message, HumanMessage)),
#             None,
#         )
#         last_ai_message = next(
#             (
#                 message
#                 for message in reversed(current_messages)
#                 if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
#             ),
#             None,
#         )

#         if last_user_message:
#             history.add_messages([last_user_message])

#         if last_ai_message:
#             history.add_messages([last_ai_message])
#             _trim_memory(history)
#             return {"response": last_ai_message}

#         fallback = AIMessage(content="Mình chưa có phản hồi phù hợp.")
#         history.add_messages([fallback])
#         _trim_memory(history)
#         return {"response": fallback}

#     graph = StateGraph(State)

#     graph.add_node("chat", chat_node)
#     graph.add_node("inject_runtime_context", inject_runtime_context_node)
#     graph.add_node("tools", tool_node)
#     graph.add_node("finalize", finalize_node)

#     graph.add_edge(START, "chat")
#     graph.add_conditional_edges(
#         "chat",
#         tools_condition,
#         {
#             "tools": "inject_runtime_context",
#             "__end__": "finalize",
#         },
#     )
#     graph.add_edge("inject_runtime_context", "tools")
#     graph.add_edge("tools", "chat")
#     graph.add_edge("finalize", END)

#     return graph.compile()
