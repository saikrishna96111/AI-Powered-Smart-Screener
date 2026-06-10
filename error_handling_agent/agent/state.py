from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class ErrorAgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    cds_source: Optional[str]
    error_text: Optional[str]
    fixed_cds: Optional[str]
    session_ended: bool
    # RAG: filled in by retrieve_docs_node and read by fix_cds_node.
    # ``references`` is the structured list (one dict per retrieved chunk),
    # ``references_text`` is the pre-rendered block that gets dropped straight
    # into the {{references}} placeholder of fix.yaml.
    references: List[Dict[str, Any]]
    references_text: Optional[str]
