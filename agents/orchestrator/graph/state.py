from typing import Any, Optional, TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage



class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    session_id: Optional[str]
    user_id: Optional[str]
    token: Optional[str]
    route: Optional[str]
    response: Optional[str]

    # pending_transaction: Optional[dict]

    # dialog_state: Optional[dict]
