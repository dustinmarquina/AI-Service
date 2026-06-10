from typing import Annotated, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class StepResult(TypedDict):
    step_id: str
    type: str
    input: dict[str, Any]
    output: Any
    status: str
    summary: str
    reasoning: NotRequired[str]


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    session_id: Optional[str]
    user_id:    Optional[str]
    token:      Optional[str]

    # planner writes one of: "execute" | "finalize" | "clarify"
    route: Optional[str]

    # plan-and-execute fields — only populated when route == "execute"
    steps:        Optional[list[dict]]       # ordered list of steps from planner
    step_index:   Optional[int]              # current step pointer
    step_results: Optional[dict[str, Any]]   # keyed by step id, stores resolved outputs
    past_steps:   Optional[list[StepResult]]
    replan_attempts: Optional[int]
    max_replan_attempts: Optional[int]

    response: Optional[str]
