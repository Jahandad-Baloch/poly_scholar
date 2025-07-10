"""
This module defines the orchestration state and dynamic context logic for PolyScholar's LangGraph workflows.

- AppState: The central state container for all static inputs, dynamic artifacts, logs, and short-term memory.
- format_dynamic_block: Helper to render a readable summary of the current state for prompt construction.

All agent nodes should treat AppState as the single source of truth for runtime facts.
"""

from typing import Any
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator

class Issue(TypedDict):
    source_agent: str
    description: str
    status: str           # OPEN | RESOLVED

class AppState(TypedDict, total=False):
    # --- static inputs ---
    topic: str
    research_question: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]

    # --- dynamic artefacts ---
    artifacts: dict[str, Any]                   # overwritten by key
    progress_log: Annotated[list[str], operator.add]
    issues_log:   Annotated[list[Issue], operator.add]
    supervisor_directives: Annotated[list[str], operator.add]
    iteration_count: int

    # --- short-term memory ---
    messages: Annotated[list, add_messages]

def format_dynamic_block(state: AppState) -> str:
    """
    Render a readable summary of the current AppState for prompt construction.
    Handles missing or incomplete state gracefully.
    """
    try:
        parts = [
            f"Research question: {state.get('research_question','—')}",
            "Inclusion: " + ", ".join(state.get('inclusion_criteria', []) or []),
            "Last 3 progress lines:\n" + "\n".join((state.get('progress_log', []) or [])[-3:]),
            "Supervisor directives:\n" + "\n".join((state.get('supervisor_directives', []) or [])[-3:]),
            f"Iteration #{state.get('iteration_count',0)}",
        ]
        return "\n".join(parts)
    except Exception as e:
        return f"[Error formatting dynamic block: {e}]"
