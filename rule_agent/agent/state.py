# agent/state.py

from typing import TypedDict, List, Dict, Optional
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[str]
    required_fields: List[str]
    collected_fields: Dict[str, str]
    missing_fields: List[str]
    explained: bool
    cds_code: Optional[str]
    approved: bool