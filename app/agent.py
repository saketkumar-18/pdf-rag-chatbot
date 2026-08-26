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
    # research-mode fields
    subqueries: list[str]       # plan decomposition of the question
    findings: list[dict]        # deduped chunks gathered across subqueries
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

    context_hits = state.get("findings") or state.get("relevant") or state.get("hits") or []
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
# Deep-research nodes (plan → research → synthesize → check)
# --------------------------------------------------------------------------
_JSON_STR_ARRAY = re.compile(r"\[[^\]]*\]")


def _parse_string_array(raw: str, max_items: int) -> list[str]:
    """Parse a model reply like ["q1","q2"] into clean subquery strings."""
    match = _JSON_STR_ARRAY.search(raw)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
    except json.JSONDecodeError:
        # tolerate unquoted items: [GPU count, BLEU score]
        inner = match.group(0)[1:-1]
        arr = [p.strip().strip("'\"") for p in inner.split(",")]
    out = [str(x).strip() for x in arr if str(x).strip()]
    return out[:max_items]


def node_plan(state: AgentState) -> dict:
    """Decompose the question into focused subqueries (1 LLM call)."""
    t0 = time.perf_counter()
    max_sub = config.RESEARCH_MAX_SUBQUERIES
    prompt = (
        "You are a research planner. Break the question below into at most "
        f"{max_sub} focused sub-questions that a document search engine can "
        "answer individually. Each sub-question must be short, concrete, and "
        "keyword-rich. If the question is already simple, return it unchanged "
        "as the only item.\n"
        'Reply with ONLY a JSON array of strings, e.g. ["sub-q1","sub-q2"].\n\n'
        f"Question: {state['question']}"
    )
    try:
        raw = llm.generate(prompt, system="You output only valid JSON arrays.")
        subs = _parse_string_array(raw, max_sub)
    except llm.LLMError:
        subs = []
    if not subs:
        subs = [state["question"]]
        detail = f"planner failed → 1 subquery (original question)"
    else:
        detail = f"{len(subs)} subqueries: " + " · ".join(s[:40] for s in subs)
    return {"subqueries": subs, "trace": _trace_entry("plan", detail, t0)}


def node_research(state: AgentState) -> dict:
    """Run one vector search per subquery; dedupe + rank the union (no LLM)."""
    t0 = time.perf_counter()
    where = {"doc_id": {"$eq": state["doc_id"]}} if state.get("doc_id") else None
    k = config.RESEARCH_K_PER_SUBQUERY
    seen: set[str] = set()
    findings: list[dict] = []
    per_sub: list[str] = []
    for sub in state.get("subqueries") or [state["question"]]:
        hits = store.search(sub, k=k, where=where)
        fresh = 0
        for h in hits:
            if h["id"] not in seen:
                seen.add(h["id"])
                findings.append(h)
                fresh += 1
        per_sub.append(f"“{sub[:34]}”→{len(hits)} (+{fresh} new)")
    findings.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    findings = findings[: config.RESEARCH_MAX_FINDINGS]
    detail = f"{len(findings)} unique chunks from {len(per_sub)} searches: " + "; ".join(per_sub)
    return {"findings": findings, "trace": _trace_entry("research", detail, t0)}


def node_synthesize(state: AgentState) -> dict:
    """Compose the final cited answer from all gathered findings (1 LLM call)."""
    t0 = time.perf_counter()
    findings = state.get("findings") or []

    if not findings:
        answer = (
            f"{_REFUSAL} in the uploaded documents to answer that. "
            "Try rephrasing, or upload documents that cover the topic."
        )
        return {
            "generation": answer,
            "verdict": "skipped",
            "trace": _trace_entry("synthesize", "refused: no findings (no LLM call)", t0),
        }

    system = prompts.SYSTEM_PROMPT
    if state.get("strict"):
        system += (
            "\n6. STRICT MODE: a previous draft was rejected for inventing "
            "facts. Every sentence MUST be supported by a context block and "
            "carry an inline [n] citation. If unsure, refuse."
        )
    prompt = prompts.build_answer_prompt(state["question"], findings)
    answer = llm.generate(prompt, system=system)
    return {
        "generation": answer,
        "trace": _trace_entry(
            "synthesize", f"{len(answer)} chars from {len(findings)} chunks", t0
        ),
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


_research_graph = None


def get_research_graph():
    """Compile the deep-research graph once, reuse forever.

    plan ─► research ─► synthesize ─► check ─┬─ grounded/skipped ─► END
                                             └─ ungrounded (1 retry) ─► synthesize (strict)
    """
    global _research_graph
    if _research_graph is None:
        if not LANGGRAPH_AVAILABLE:
            raise AgentUnavailable("langgraph is not installed.")
        builder = StateGraph(AgentState)
        builder.add_node("plan", node_plan)
        builder.add_node("research", node_research)
        builder.add_node("synthesize", node_synthesize)
        builder.add_node("check", node_check_answer)
        builder.add_node("retry_gate", _enter_retry)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "research")
        builder.add_edge("research", "synthesize")
        builder.add_edge("synthesize", "check")
        builder.add_conditional_edges(
            "check", route_after_check, {"generate": "retry_gate", END: END}
        )
        builder.add_edge("retry_gate", "synthesize")
        _research_graph = builder.compile()
    return _research_graph


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


def run_agent(
    question: str,
    top_k: int = 5,
    doc_id: str | None = None,
    mode: str = "agent",
) -> dict[str, Any]:
    """Execute a graph. Returns answer + trace + diagnostics.

    mode="agent"    → CRAG self-correcting pipeline (retrieve→grade→generate→check)
    mode="research" → deep-research pipeline (plan→research→synthesize→check)
    """
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

    if mode == "research":
        final_state = get_research_graph().invoke(initial)
        cited = final_state.get("findings") or []
        all_hits = cited
    else:
        final_state = get_graph().invoke(initial)
        cited = final_state.get("relevant") or []
        all_hits = final_state.get("hits") or []

    return {
        "answer": final_state.get("generation", ""),
        "cited_hits": cited or all_hits,  # blocks the answer was built from
        "all_hits": all_hits,
        "trace": final_state.get("trace", []),
        "rewrites": final_state.get("rewrites", 0),
        "retries": final_state.get("retries", 0),
        "verdict": final_state.get("verdict", "grounded"),
        "mode": mode,
        "subqueries": final_state.get("subqueries", []),
    }
