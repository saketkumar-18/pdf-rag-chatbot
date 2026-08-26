"""Tests for the LangGraph agentic RAG pipeline (LLM mocked, no network).

Run:  .venv/Scripts/python.exe -m pytest tests/test_agent.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
def _fake_hits(n: int = 2, score: float = 0.6) -> list[dict]:
    return [
        {
            "id": f"doc_test_c{i}",
            "text": f"Chunk {i}: the transformer uses scaled dot-product attention [softmax].",
            "metadata": {
                "source": "paper.pdf",
                "doc_id": "doc_test",
                "page_start": i + 1,
                "page_end": i + 1,
                "chunk_index": i,
            },
            "distance": round(1 - score, 4),
            "score": score,
        }
        for i in range(n)
    ]


class MockLLM:
    """Dispatches canned replies by prompt fingerprint; records every call."""

    def __init__(self, grade_reply=None, check_reply="yes"):
        self.calls: list[str] = []
        self.grade_reply = grade_reply
        self.check_reply = check_reply

    def __call__(self, prompt: str, system: str | None = None, model: str | None = None) -> str:
        self.calls.append(prompt)
        if "relevance grader" in prompt:
            return self.grade_reply if self.grade_reply is not None else '["yes", "yes"]'
        if "Rewrite the question" in prompt:
            return "scaled dot-product attention formula softmax"
        if "answer quality grader" in prompt:
            return self.check_reply
        # answer generation
        return "The model uses scaled dot-product attention with softmax [1][2]."


@pytest.fixture
def patched(monkeypatch):
    """Patch store.search + llm.generate; yield the mock for assertions."""
    from app import agent, llm, store

    mock = MockLLM()
    monkeypatch.setattr(store, "search", lambda q, k=None, where=None: _fake_hits(2))
    monkeypatch.setattr(llm, "generate", mock)
    monkeypatch.setattr(agent.config, "AGENT_MAX_REWRITES", 1)
    monkeypatch.setattr(agent.config, "AGENT_CHECK_ANSWER", True)
    yield mock


# ----------------------------------------------------------------------
# Unit: helpers
# ----------------------------------------------------------------------
def test_parse_yes_no_array_strict():
    from app.agent import _parse_yes_no_array

    assert _parse_yes_no_array('["yes","no","yes"]', 3) == ["yes", "no", "yes"]
    assert _parse_yes_no_array('noise ["yes", "no"] noise', 2) == ["yes", "no"]
    # single quotes / bare words tolerated
    assert _parse_yes_no_array("['yes', no]", 2) == ["yes", "no"]
    # wrong length or junk -> None
    assert _parse_yes_no_array('["yes"]', 2) is None
    assert _parse_yes_no_array('["yes","maybe"]', 2) is None
    assert _parse_yes_no_array("no array here", 2) is None


def test_resolve_mode(patched, monkeypatch):
    from app import agent

    monkeypatch.setattr(agent.config, "AGENT_MODE", "auto")
    assert agent.resolve_mode(None) is True      # auto + LLM present
    assert agent.resolve_mode(False) is False    # explicit opt-out wins

    monkeypatch.setattr(agent.config, "AGENT_MODE", "off")
    assert agent.resolve_mode(None) is False
    assert agent.resolve_mode(True) is False

    monkeypatch.setattr(agent.config, "AGENT_MODE", "on")
    assert agent.resolve_mode(None) is True


# ----------------------------------------------------------------------
# Graph: happy path
# ----------------------------------------------------------------------
def test_happy_path(patched):
    from app import agent

    result = agent.run_agent("What attention mechanism is used?", top_k=2)
    nodes = [s["node"] for s in result["trace"]]
    assert nodes == ["retrieve", "grade", "generate", "check"]
    assert result["rewrites"] == 0
    assert result["retries"] == 0
    assert result["verdict"] == "grounded"
    assert "[1]" in result["answer"]
    # 3 LLM calls: grade + generate + check
    assert len(patched.calls) == 3


# ----------------------------------------------------------------------
# Graph: rewrite loop when nothing is relevant
# ----------------------------------------------------------------------
def test_rewrite_loop_then_success(patched):
    from app import agent

    # first grading pass rejects everything, second accepts
    replies = iter(['["no", "no"]', '["yes", "yes"]'])
    patched.grade_reply = None

    def grading(prompt, system=None, model=None):
        patched.calls.append(prompt)
        if "relevance grader" in prompt:
            return next(replies)
        if "Rewrite the question" in prompt:
            return "better query keywords"
        if "answer quality grader" in prompt:
            return "yes"
        return "Answer with citations [1]."

    from app import llm
    import app.agent as agent_mod

    # rebind generate inside the agent module's namespace via llm module
    original = llm.generate
    llm.generate = grading
    try:
        result = agent_mod.run_agent("obscure question", top_k=2)
    finally:
        llm.generate = original

    nodes = [s["node"] for s in result["trace"]]
    assert nodes == [
        "retrieve", "grade", "rewrite", "retrieve", "grade", "generate", "check"
    ]
    assert result["rewrites"] == 1
    assert result["verdict"] == "grounded"


def test_exhausted_rewrites_refuses_without_llm_answer(patched, monkeypatch):
    from app import agent, llm

    patched.grade_reply = '["no", "no"]'  # always irrelevant
    result = agent.run_agent("unanswerable question", top_k=2)
    nodes = [s["node"] for s in result["trace"]]
    assert nodes == [
        "retrieve", "grade", "rewrite", "retrieve", "grade", "generate", "check"
    ]
    assert "don't have enough information" in result["answer"]
    assert result["verdict"] == "skipped"
    # LLM was used for grade x2 + rewrite only — never for generation
    assert not any("Context blocks" in c for c in patched.calls)


# ----------------------------------------------------------------------
# Graph: self-correction on hallucination
# ----------------------------------------------------------------------
def test_self_correction_on_ungrounded_answer(patched):
    from app import agent, llm

    check_replies = iter(["no", "yes"])  # first draft rejected, second accepted
    systems_seen: list[str] = []

    def gen(prompt, system=None, model=None):
        patched.calls.append(prompt)
        if system:
            systems_seen.append(system)
        if "relevance grader" in prompt:
            return '["yes", "yes"]'
        if "answer quality grader" in prompt:
            return next(check_replies)
        return "Draft answer [1]."

    original = llm.generate
    llm.generate = gen
    try:
        result = agent.run_agent("question", top_k=2)
    finally:
        llm.generate = original

    nodes = [s["node"] for s in result["trace"]]
    assert nodes == [
        "retrieve", "grade", "generate", "check",
        "retry_gate", "generate", "check",
    ]
    assert result["retries"] == 1
    assert result["verdict"] == "grounded"
    # strict mode reminder was injected into the system prompt on the retry
    assert any("STRICT MODE" in s for s in systems_seen)


def test_ungrounded_twice_gives_up(patched):
    from app import agent, llm

    def gen(prompt, system=None, model=None):
        patched.calls.append(prompt)
        if "relevance grader" in prompt:
            return '["yes", "yes"]'
        if "answer quality grader" in prompt:
            return "no"  # always rejects
        return "Hallucinated answer [1]."

    original = llm.generate
    llm.generate = gen
    try:
        result = agent.run_agent("question", top_k=2)
    finally:
        llm.generate = original

    assert result["retries"] == 1          # only one retry allowed
    assert result["verdict"] == "ungrounded"
    nodes = [s["node"] for s in result["trace"]]
    assert nodes.count("generate") == 2


# ----------------------------------------------------------------------
# Degradation: unparseable grading falls back to heuristic
# ----------------------------------------------------------------------
def test_grade_fallback_on_garbage(patched):
    from app import agent

    patched.grade_reply = "I cannot output JSON, sorry!"
    result = agent.run_agent("question", top_k=2)
    grade_step = next(s for s in result["trace"] if s["node"] == "grade")
    assert "heuristic" in grade_step["detail"]
    assert result["answer"]  # still produced an answer


# ----------------------------------------------------------------------
# API integration: /api/ask returns agent trace
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def isolated_storage_module(tmp_path_factory):
    import app.config as config

    tmp = tmp_path_factory.mktemp("agent_api_data")
    config.CHROMA_DIR = tmp / "chroma"
    config.UPLOADS_DIR = tmp / "uploads"
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    yield


def test_api_ask_agent_mode(isolated_storage_module, patched, monkeypatch):
    from fastapi.testclient import TestClient

    import app.config as config
    from app.main import app

    monkeypatch.setattr(config, "AGENT_MODE", "auto")

    client = TestClient(app)

    # health exposes agent availability
    health = client.get("/api/health").json()
    assert health["agent"]["available"] is True
    assert health["agent"]["langgraph"]

    # agent ask
    r = client.post("/api/ask", json={"question": "What attention is used?", "agent": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["agent"]["enabled"] is True
    assert data["agent"]["trace"], "trace must not be empty"
    assert data["agent"]["verdict"] == "grounded"
    assert data["citations"], "citations parsed from [n] markers"
    assert data["sources"]

    # explicit opt-out -> classic path, no agent block
    r2 = client.post("/api/ask", json={"question": "What attention is used?", "agent": False})
    assert r2.status_code == 200
    assert "agent" not in r2.json()
