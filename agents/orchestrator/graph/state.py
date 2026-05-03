from typing import Any, Optional, TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage

class TaskState(TypedDict, total=False):
    intent: str

    required_slots: list[str]
    filled_slots: dict[str, Any]
    missing_slots: list[str]

    status: str  # collecting | ready | executing | done
    next_action: str  # ask_user | execute | clarify

    current_step: str


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    task_state: Optional[TaskState]

    tool_input: Optional[dict]
    tool_output: Optional[Any]

    sql_results: Optional[Any]
    sql_query: Optional[str]
    sql_error: Optional[str]

    retrieved_docs: Optional[list[str]]

    ui_actions: Optional[dict]

    response: Optional[str]

    # pending_transaction: Optional[dict]

    # dialog_state: Optional[dict]