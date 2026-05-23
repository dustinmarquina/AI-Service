import json
import base64

from fastapi import Depends, APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from models.schemas import ChatRequest

security = HTTPBearer()

router = APIRouter(prefix="/api/chat", tags=["Chat"])


def _extract_user_id_from_token(token: str | None) -> str | None:
    raw = str(token or "").strip()
    if not raw:
        return None

    parts = raw.split(".")
    if len(parts) != 3:
        return None

    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None

    sub = payload.get("sub")
    return sub.strip() if isinstance(sub, str) and sub.strip() else None


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)
    return str(content or "")


def _extract_stream_text(update: dict) -> str | None:
    interrupts = update.get("__interrupt__")
    if interrupts:
        for interrupt in interrupts:
            value = getattr(interrupt, "value", None)
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
            elif isinstance(value, dict):
                text = str(value.get("text", "")).strip()
                if text:
                    return text

    for payload in update.values():
        if not isinstance(payload, dict):
            continue

        response = payload.get("response")
        if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None):
            text = _message_text(response).strip()
            if text:
                return text

        for message in payload.get("messages", []):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                text = _message_text(message).strip()
                if text:
                    return text

    return None


def _sse_text_event(text: str) -> str:
    return f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"


@router.post("/message")
async def chat_message(
    request_body: ChatRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        # Access the orchestrator main graph from app state
        graph = request.app.state.main_graph
        token = credentials.credentials if credentials else None
        user_id = _extract_user_id_from_token(token)
        print(f"Received chat message: {request_body.message}, token: {token}")

        payload = {
            "messages": [HumanMessage(content=request_body.message)],
            "user_id": user_id,
            "token": token,
            "session_id": "<SESSION_ID>",  # fix the session_id for now
        }
        config = {
            "configurable": {
                "thread_id": payload["session_id"],
            }
        }

        snapshot = await graph.aget_state(config)
        has_pending_interrupt = bool(getattr(snapshot, "interrupts", ()))
        graph_input = Command(resume=request_body.message) if has_pending_interrupt else payload

        async def event_stream():
            last_emitted = None
            async for update in graph.astream(
                graph_input,
                config=config,
                stream_mode="updates",
            ):
                text = _extract_stream_text(update)
                if not text or text == last_emitted:
                    continue
                last_emitted = text
                yield _sse_text_event(text)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
