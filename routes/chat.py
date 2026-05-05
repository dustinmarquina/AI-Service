from fastapi import Depends, APIRouter, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.messages import HumanMessage
from models.schemas import ChatRequest
security = HTTPBearer()

router = APIRouter(prefix="/api/chat", tags=["Chat"])


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
        print(f"Received chat message: {request_body.message}, token: {token}")

        result = await graph.ainvoke({
            "messages": [HumanMessage(content=request_body.message)],
            "user_id": "<USER_ID>", # fix the user_id for now
            "token": token,
            "session_id": "<SESSION_ID>", # fix the session_id for now
        })

        response = result.get("response")
        if response is None and result.get("messages"):
            last_message = result["messages"][-1]
            response = last_message.content if hasattr(last_message, "content") else str(last_message)

        return {
            "response": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))