from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from models.schemas import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/message")
async def chat_message(request_body: ChatRequest, request: Request):
    try:
        # Access the orchestrator main graph from app state
        graph = request.app.state.main_graph

        result = await graph.ainvoke({
            "messages": [HumanMessage(content=request_body.message)]
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