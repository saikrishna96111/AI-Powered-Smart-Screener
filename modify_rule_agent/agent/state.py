from typing import List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class ModifyRuleState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    modify_flow_started: bool
    cds_original: Optional[str]
    cds_working: Optional[str]
    summary_sent: bool
    session_ended: bool
