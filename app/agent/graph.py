"""The LangGraph corrective-RAG agent: plan → retrieve (parallel) → grade → synthesize.

Agent strategy (state-of-the-art corrective RAG on LangGraph):
  - PLAN node:     LLM decides which of the three retrievers (vector/SQL/graph) to invoke.
  - RETRIEVE node: executes all planned retriever calls concurrently.
  - GRADE node:    LLM reflects on evidence sufficiency.
  - Loop:          insufficient evidence re-enters PLAN with accumulated history
                   (bounded by MAX_AGENT_ITERATIONS).
  - SYNTHESIZE:    grounded final answer with inline document citations.
"""

import json
import logging
import re
from typing import Literal

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.agent.prompts import GRADE_PROMPT, PLAN_PROMPT, SYNTHESIZE_PROMPT
from app.agent.state import AgentState
from app.agent.tools import RetrieverToolkit
from app.config import get_settings

logger = logging.getLogger(__name__)


def _json_call(client: OpenAI, model: str, prompt: str) -> dict:
    """Chat call that expects strict JSON; tolerates markdown fences."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = (resp.choices[0].message.content or "{}").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _format_evidence(evidence: list[dict]) -> str:
    parts = []
    for i, ev in enumerate(evidence, 1):
        if "error" in ev:
            parts.append(f"[{i}] {ev['source']} ERROR: {ev['error']}")
            continue
        payload = json.dumps(ev.get("payload", []), default=str, ensure_ascii=False)
        if len(payload) > 3000:
            payload = payload[:3000] + " ...[truncated]"
        parts.append(f"[{i}] {ev['source'].upper()} query: {ev.get('query', '')}\n{payload}")
    return "\n\n".join(parts) if parts else "(no evidence yet)"


class DocGraphAgent:
    """Builds and runs the LangGraph agent with a configurable OpenAI backbone."""

    def __init__(self, toolkit: RetrieverToolkit | None = None) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — the agent requires it.")
        self.settings = s
        self.client = OpenAI(api_key=s.openai_api_key, timeout=s.openai_timeout)
        self.model = s.openai_model
        self.toolkit = toolkit or RetrieverToolkit()
        self.graph = self._build_graph()

    # ── Node implementations ──────────────────────────────────────

    def _plan_node(self, state: AgentState) -> dict:
        question = state["question"]
        iteration = state.get("iteration", 0)

        history = ""
        if state.get("plan_notes"):
            notes = state["plan_notes"][-self.settings.max_agent_iterations :]
            evidence_summary = _format_evidence(state.get("evidence", []))
            history = (
                f"Rounds so far: {len(notes)}.\n"
                f"Plans: {json.dumps(notes, ensure_ascii=False)}\n"
                f"Evidence so far:\n{evidence_summary[:4000]}"
            )

        data = _json_call(
            self.client, self.model, PLAN_PROMPT.format(question=question, history=history)
        )
        retrievers = data.get("retrievers", [])
        notes = data.get("notes", "")
        logger.info("[plan] round=%d retrievers=%s notes=%s", iteration + 1,
                    [r.get("tool") for r in retrievers], notes)
        return {
            "iteration": iteration + 1,
            "retrievers": retrievers,
            "plan_notes": [notes],
        }

    def _retrieve_node(self, state: AgentState) -> dict:
        plan = state.get("retrievers", []) or []
        results = [self.toolkit.execute(entry) for entry in plan]
        logger.info("[retrieve] executed %d retriever calls", len(results))
        return {"evidence": results}

    def _grade_node(self, state: AgentState) -> dict:
        # Bound the correction loop.
        if state.get("iteration", 0) >= self.settings.max_agent_iterations:
            logger.info("[grade] iteration cap reached; proceeding to synthesis")
            return {"grade": "sufficient", "grade_notes": "iteration cap reached"}

        data = _json_call(
            self.client,
            self.model,
            GRADE_PROMPT.format(
                question=state["question"], evidence=_format_evidence(state.get("evidence", []))
            ),
        )
        grade = data.get("grade", "insufficient")
        if grade not in ("sufficient", "insufficient"):
            grade = "insufficient"
        logger.info("[grade] %s — %s", grade, data.get("notes", ""))
        return {"grade": grade, "grade_notes": data.get("notes", "")}

    def _synthesize_node(self, state: AgentState) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": SYNTHESIZE_PROMPT.format(
                        question=state["question"],
                        evidence=_format_evidence(state.get("evidence", [])),
                    ),
                }
            ],
            temperature=0.1,
        )
        answer = resp.choices[0].message.content or ""
        logger.info("[synthesize] answer length=%d", len(answer))
        return {"answer": answer}

    def _route_after_grade(self, state: AgentState) -> Literal["plan", "synthesize"]:
        if state.get("grade") == "sufficient":
            return "synthesize"
        return "plan"

    # ── Graph construction ────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade", self._grade_node)
        builder.add_node("synthesize", self._synthesize_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"plan": "plan", "synthesize": "synthesize"},
        )
        builder.add_edge("synthesize", END)
        return builder.compile()

    # ── Public API ────────────────────────────────────────────────

    def run(self, question: str) -> dict:
        """Run the agent; returns {"answer", "evidence", "iterations", "plans"}."""
        final: AgentState = self.graph.invoke(
            {"question": question, "iteration": 0, "evidence": []}
        )
        return {
            "answer": final.get("answer", ""),
            "iterations": final.get("iteration", 0),
            "plans": final.get("plan_notes", []),
            "evidence_sources": [e.get("source") for e in final.get("evidence", [])],
        }
