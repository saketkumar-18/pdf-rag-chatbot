"""Agentic RAG powered by LangGraph (CRAG-style self-correcting pipeline).

Graph
-----
    START ─► retrieve ─► grade_documents ─┬─ relevant docs ──────────► generate ─► check_answer ─► END
                                          │                                        │ (not grounded)
                                          ├─ nothing relevant, rewrites left ─► rewrite_query ─┐
                                          │                                                    │
                                          └─ nothing relevant, no rewrites ─► generate (refuse)│
                                                                                               │
    rewrite_query ─► retrieve (loop, max RAG_AGENT_MAX_REWRITES)          generate ◄───────────┘
                                                                              (max 1 retry)

Design rules
------------
* ONE LLM call per node (grading batches all chunks into a single JSON call)
  so the whole run stays inside serverless timeouts: typical = 3 calls.
* LLM calls go through the existing stdlib backend in ``app/llm.py``
  (Ollama / OpenAI-compatible / HF router) — LangGraph orchestrates,
  langchain_core builds the prompts, no new network dependency.
* Every node appends a ``trace`` entry so the UI can show the agent's
  reasoning timeline (retrieved → graded → rewrote → generated → checked).
* Graceful degradation: if langgraph isn't installed or the LLM backend is
  the built-in extractive engine, ``resolve_mode`` reports the agent as
  unavailable and the API serves the classic single-shot RAG path.
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from . import config, llm, prompts, store

# --------------------------------------------------------------------------
# Optional dependency probe (serverless bundles may omit langgraph)
# --------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import langgraph.version as _lg_version
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_VERSION = _lg_version.__version__
    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGGRAPH_VERSION = None
    LANGGRAPH_AVAILABLE = False


class AgentUnavailable(RuntimeError):
    """Raised when the agentic path cannot run (missing dep / no LLM)."""


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    question: str
    effective_query: str
    doc_id: str | None
    top_k: int
    hits: list[dict]            # raw retrieval results (latest retrieval)
    relevant: list[dict]        # chunks that passed grading
    generation: str
    strict: bool                # regenerate with a stricter reminder
    rewrites: int
    retries: int
    verdict: str                # "grounded" | "ungrounded" | "skipped"
    # reducer: each node returns {"trace": [entry]} and entries accumulate
    trace: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _trace_entry(node: str, detail: str, t0: float) -> list[dict]:
    """One trace entry; nodes return {"trace": [...]} and the reducer appends."""
    return [
        {"node": node, "detail": detail, "ms": int((time.perf_counter() - t0) * 1000)}
    ]


_JSON_ARRAY = re.compile(r"\[[^\]]*\]")


def _parse_yes_no_array(raw: str, expected: int) -> list[str] | None:
    """Parse a model reply like ["yes","no","yes"] tolerantly."""
    match = _JSON_ARRAY.search(raw)
    if not match:
        return None
    try:
        arr = json.loads(match.group(0))
    except json.JSONDecodeError:
        # tolerate single quotes / bare words: ["yes", no, 'yes']
        words = re.findall(r"yes|no", match.group(0), flags=re.IGNORECASE)
        arr = [w.lower() for w in words]
    arr = [str(x).strip().lower() for x in arr]
    if len(arr) != expected:
        return None
    if not all(x in {"yes", "no"} for x in arr):
        return None
    return arr


def _heuristic_grade(hits: list[dict]) -> list[dict]:
    """Score-floor fallback used when no LLM grading is possible."""
    floor = 0.25
    kept = [h for h in hits if h.get("score", 0.0) >= floor]
    return kept or hits[:2]


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def node_retrieve(state: AgentState) -> dict:
    t0 = time.perf_counter()
    query = state.get("effective_query") or state["question"]
    where = {"doc_id": {"$eq": state["doc_id"]}} if state.get("doc_id") else None
    hits = store.search(query, k=state.get("top_k", config.TOP_K), where=where)
    detail = (
        f"{len(hits)} chunks for “{query[:60]}”"
        + (f", best score {hits[0]['score']}" if hits else "")
    )
    return {"hits": hits, "trace": _trace_entry("retrieve", detail, t0)}


def node_grade_documents(state: AgentState) -> dict:
    t0 = time.perf_counter()
    hits = state.get("hits") or []
    if not hits:
        return {"relevant": [], "trace": _trace_entry("grade", "no chunks retrieved", t0)}

    backend = config.effective_llm_backend()
    if backend == "extractive":
        kept = _heuristic_grade(hits)
        return {
            "relevant": kept,
            "trace": _trace_entry("grade", f"heuristic: kept {len(kept)}/{len(hits)}", t0),
        }

    blocks = prompts.build_context_block(hits)
    prompt = (
        "You are a relevance grader for a retrieval system.\n"
        f"Question: {state['question']}\n\n"
        "For each numbered context block below, decide whether it contains "
        "information that could help answer the question.\n"
        'Reply with ONLY a JSON array of "yes"/"no" strings, one per block, '
        'e.g. ["yes","no","yes"].\n\n'
        f"Blocks:\n{blocks}"
    )
    try:
        raw = llm.generate(prompt, system="You output only valid JSON arrays.")
    except llm.LLMError:
        kept = _heuristic_grade(hits)
        return {
            "relevant": kept,
            "trace": _trace_entry("grade", f"LLM failed → heuristic kept {len(kept)}/{len(hits)}", t0),
        }

    verdicts = _parse_yes_no_array(raw, expected=len(hits))
    if verdicts is None:
        kept = _heuristic_grade(hits)
        return {
            "relevant": kept,
            "trace": _trace_entry("grade", f"unparseable reply → heuristic kept {len(kept)}/{len(hits)}", t0),
        }

    kept = [h for h, v in zip(hits, verdicts) if v == "yes"]
    return {
        "relevant": kept,
        "trace": _trace_entry("grade", f"kept {len(kept)}/{len(hits)} chunks", t0),
    }


def node_rewrite_query(state: AgentState) -> dict:
    t0 = time.perf_counter()
    prompt = (
        "Rewrite the question below into a better search query for a vector "
        "database of documents. Keep it short, concrete, and keyword-rich. "
        "Output ONLY the rewritten query — no quotes, no explanations.\n\n"
        f"Question: {state['question']}"
    )
    try:
        rewritten = llm.generate(prompt, system="You rewrite search queries.").strip()
    except llm.LLMError:
        rewritten = ""

    if not rewritten or len(rewritten) > 400:
        # cheap deterministic fallback: drop question words, keep content terms
        rewritten = re.sub(r"[^\w\s]", " ", state["question"])

    return {
        "effective_query": rewritten,
        "rewrites": state.get("rewrites", 0) + 1,
        "trace": _trace_entry("rewrite", f"new query: “{rewritten[:60]}”", t0),
    }


_REFUSAL = "I don't have enough information"


def node_generate(state: AgentState) -> dict:
    t0 = time.perf_counter()
    relevant = state.get("relevant") or []

    if not relevant:
        # Nothing passed grading and rewrites are exhausted -> honest refusal,
        # no LLM call needed.
        answer = (
            f"{_REFUSAL} in the uploaded documents to answer that. "
            "Try rephrasing, or upload documents that cover the topic."
        )
        return {
            "generation": answer,
            "verdict": "skipped",
            "trace": _trace_entry("generate", "refused: no relevant context (no LLM call)", t0),
        }

    context_hits = relevant
    system = prompts.SYSTEM_PROMPT
    if state.get("strict"):
        system += (
            "\n6. STRICT MODE: a previous draft was rejected for inventing "
            "facts. Every sentence MUST be supported by a context block and "
            "carry an inline [n] citation. If unsure, refuse."
        )
    prompt = prompts.build_answer_prompt(state["question"], context_hits)
    try:
        answer = llm.generate(prompt, system=system)
    except llm.LLMError as exc:
        raise
    return {
        "generation": answer,
        "trace": _trace_entry("generate", f"{len(answer)} chars from {len(context_hits)} chunks", t0),
    }


def node_check_answer(state: AgentState) -> dict:
    t0 = time.perf_counter()
    answer = state.get("generation", "")

    if state.get("verdict") == "skipped" or _REFUSAL in answer:
        # Refusals are safe by construction — accept without an LLM call,
        # and keep the "skipped" verdict so the UI can show no LLM was used.
        return {
            "verdict": "skipped",
            "trace": _trace_entry("check", "refusal accepted (no grounding check needed)", t0),
        }

    if not config.AGENT_CHECK_ANSWER:
        return {"verdict": "grounded", "trace": _trace_entry("check", "disabled by config", t0)}

    context_hits = state.get("relevant") or state.get("hits") or []
    prompt = (
        "You are an answer quality grader. Decide whether the answer stays "
        "grounded in the context: it must not invent facts, numbers, or names "
        "that are absent from the context blocks.\n\n"
        f"Question: {state['question']}\n\n"
        f"Context:\n{prompts.build_context_block(context_hits)}\n\n"
        f"Answer:\n{answer}\n\n"
        'Reply with ONLY "yes" (grounded) or "no" (invented content).'
    )
    try:
        raw = llm.generate(prompt, system="You answer only yes or no.").strip().lower()
    except llm.LLMError:
        return {
            "verdict": "grounded",
            "trace": _trace_entry("check", "grader LLM failed → accepting answer", t0),
        }

    grounded = raw.startswith("yes")
    return {
        "verdict": "grounded" if grounded else "ungrounded",
        "trace": _trace_entry("check", "grounded ✓" if grounded else "hallucination detected ✗", t0),
    }


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def route_after_grade(state: AgentState) -> str:
    if state.get("relevant"):
        return "generate"
    if state.get("rewrites", 0) < config.AGENT_MAX_REWRITES:
        return "rewrite"
    return "generate"  # exhausted -> generate will refuse cleanly


def route_after_check(state: AgentState) -> str:
    if state.get("verdict") == "ungrounded" and state.get("retries", 0) < 1:
        return "generate"
    return END


def _enter_retry(state: AgentState) -> dict:
    """Bump retry counter + strict flag when looping back into generate."""
    t0 = time.perf_counter()
    return {
        "retries": state.get("retries", 0) + 1,
        "strict": True,
        "trace": _trace_entry("retry_gate", "regenerating in strict mode", t0),
    }


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------
_graph = None


def get_graph():
    """Compile once, reuse forever (thread-safe: state is per-invocation)."""
    global _graph
    if _graph is None:
        if not LANGGRAPH_AVAILABLE:
            raise AgentUnavailable("langgraph is not installed.")
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", node_retrieve)
        builder.add_node("grade", node_grade_documents)
        builder.add_node("rewrite", node_rewrite_query)
        builder.add_node("generate", node_generate)
        builder.add_node("check", node_check_answer)
        builder.add_node("retry_gate", _enter_retry)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"}
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", "check")
        builder.add_conditional_edges(
            "check", route_after_check, {"generate": "retry_gate", END: END}
        )
        builder.add_edge("retry_gate", "generate")
        _graph = builder.compile()
    return _graph


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def available() -> bool:
    """Agent can run: langgraph installed AND a real LLM backend configured."""
    return (
        LANGGRAPH_AVAILABLE
        and config.effective_llm_backend() != "extractive"
    )


def resolve_mode(requested: bool | None) -> bool:
    """Decide whether a given /api/ask request uses the agentic path."""
    if config.AGENT_MODE == "off":
        return False
    if requested is False:
        return False
    if config.AGENT_MODE == "on":
        return requested is True or available()
    # auto (default): use the agent whenever it can actually run
    return available()


def run_agent(question: str, top_k: int = 5, doc_id: str | None = None) -> dict[str, Any]:
    """Execute the full graph. Returns answer + trace + diagnostics."""
    if not LANGGRAPH_AVAILABLE:
        raise AgentUnavailable("langgraph is not installed.")
    if config.effective_llm_backend() == "extractive":
        raise AgentUnavailable("No LLM backend configured (extractive mode).")

    initial: AgentState = {
        "question": question.strip(),
        "effective_query": question.strip(),
        "doc_id": doc_id,
        "top_k": top_k,
        "rewrites": 0,
        "retries": 0,
        "strict": False,
        "trace": [],
    }
    final_state = get_graph().invoke(initial)

    relevant = final_state.get("relevant") or []
    hits = final_state.get("hits") or []
    return {
        "answer": final_state.get("generation", ""),
        "cited_hits": relevant or hits,   # blocks the answer was built from
        "all_hits": hits,
        "trace": final_state.get("trace", []),
        "rewrites": final_state.get("rewrites", 0),
        "retries": final_state.get("retries", 0),
        "verdict": final_state.get("verdict", "grounded"),
    }
