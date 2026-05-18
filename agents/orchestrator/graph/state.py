from typing import Annotated, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class StepResult(TypedDict):
    step_id: str
    type: str        # "sql" | "tool"
    output: Any      # raw result for placeholder resolution
    summary: str     # human-readable Vietnamese summary for final response


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    session_id: Optional[str]
    user_id:    Optional[str]
    token:      Optional[str]

    # planner writes one of: "sql_agent" | "chat" | "execute" | "finalize" | "clarify"
    route: Optional[str]

    # plan-and-execute fields — only populated when route == "execute"
    steps:        Optional[list[dict]]       # ordered list of steps from planner
    step_index:   Optional[int]              # current step pointer
    step_results: Optional[dict[str, Any]]   # keyed by step id, stores resolved outputs

    response: Optional[str]