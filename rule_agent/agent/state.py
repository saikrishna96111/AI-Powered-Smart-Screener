# agent/state.py

from typing import TypedDict, List, Dict, Optional
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


def merge_collected_fields(
    left: Optional[Dict[str, str]], right: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """LangGraph reducer — merges incremental collected_fields updates into session dict."""
    merged = dict(left or {})
    if right:
        merged.update(dict(right))
    return merged


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    cds_flow_started: bool
    intent: Optional[str]
    # None = not planned yet; [] = planned, no clarifications needed
    required_fields: Optional[List[str]]
    collected_fields: Annotated[Dict[str, str], merge_collected_fields]
    missing_fields: List[str]
    explained: bool
    cds_code: Optional[str]
    cds_review_done: bool
    approved: bool
    cds_delivered: bool
    session_ended: bool
    # Companion artifacts emitted alongside the CDS view (baseinfo JSON, abapGit XML)
    cds_ddl_name: Optional[str]
    cds_baseinfo: Optional[str]
    cds_xml: Optional[str]
    cds_artifacts_dir: Optional[str]