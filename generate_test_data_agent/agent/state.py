from typing import List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class GenerateTestDataState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    cds_source: Optional[str]
    srvd_source: Optional[str]
    rule_context: Optional[str]
    session_ended: bool
