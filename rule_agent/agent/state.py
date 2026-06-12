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
    # Companion artifacts emitted alongside the CDS view (parameters JSON plus the
    # RAP service-binding files: .srvd.srvd source for the service definition and
    # .srvb.srvb XML for the OData V4 / UI binding).
    cds_ddl_name: Optional[str]
    cds_parameters_text: Optional[str]
    cds_service_def_name: Optional[str]
    cds_service_def_text: Optional[str]
    cds_service_binding_name: Optional[str]
    cds_service_binding_text: Optional[str]
    cds_artifacts_dir: Optional[str]
    # CDS view parameter collection (runs after required_fields are filled).
    # params_phase walks the user through: None -> "ask_date" -> "ask_more" -> "done"
    # cds_parameter_inputs stores the raw user answers for the parameter questions.
    # cds_parameters is the structured list parsed out of the generated CDS
    # (name + type + label) that ends up in the .parameters file.
    params_phase: Optional[str]
    cds_parameter_inputs: Dict[str, str]
    cds_parameters: List[Dict[str, str]]
    parameters_collection_done: bool
    # Index of HumanMessages at the moment the recap was sent. approval_node only
    # processes user input AFTER this counter advances (i.e. the user has actually
    # replied to the recap, not the last clarification "yes").
    summary_human_count: int
    # Self-healing CDS syntax review (runs between cds and cds_review).
    cds_syntax_status: Optional[str]   # "PASSED", "FAILED", or "ERROR"
    cds_syntax_issues: List[str]
    cds_syntax_retries: int            # number of fix attempts actually made
    cds_syntax_review_done: bool
    # Reference examples + supporting excerpts retrieved from the shared Chroma
    # store (built by error_handling_agent/scripts/build_index.py). Populated
    # by retrieve_examples_node right before cds_node.
    reference_examples_text: Optional[str]
    reference_examples_meta: List[Dict[str, str]]