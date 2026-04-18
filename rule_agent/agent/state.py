# agent/state.py

from typing import TypedDict, List, Dict, Optional
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[str]
    # None = not planned yet; [] = planned, no clarifications needed
    required_fields: Optional[List[str]]
    collected_fields: Dict[str, str]
    missing_fields: List[str]
    explained: bool
    cds_code: Optional[str]
    cds_review_done: bool
    approved: bool
    cds_delivered: bool
    session_ended: bool