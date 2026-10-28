"""LangGraph agent state."""

from typing import Annotated, Any, Literal, TypedDict


def append_list(current: list, update: list) -> list:
    return (current or []) + (update or [])


class AgentState(TypedDict, total=False):
    """State for the corrective-RAG agent graph."""
    # Input
    question: str
    # Loop control
    iteration: int
    # Planning: which retrievers to use this round
    retrievers: Annotated[list[str], append_list]      # last plan wins in `retrievers`
    plan_notes: Annotated[list[str], append_list]      # LLM's reasoning per round
    # Evidence gathered (each item: {"source": "vector|sql|graph", "payload": ...})
    evidence: Annotated[list[dict[str, Any]], append_list]
    # Grading
    grade: Literal["sufficient", "insufficient"]
    grade_notes: str
    # Output
    answer: str
    error: str
