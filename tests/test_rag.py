"""End-to-end tests: chunking, vector store, and the full RAG API.

Run:  .venv/Scripts/python.exe -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_storage(tmp_path_factory):
    """Point config at temp dirs BEFORE any module touches storage."""
    import app.config as config

    tmp = tmp_path_factory.mktemp("rag_data")
    config.CHROMA_DIR = tmp / "chroma"
    config.UPLOADS_DIR = tmp / "uploads"
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(scope="session")
def demo_pdf(tmp_path_factory):
    """Build the 3-page demo PDF described in scripts/make_demo_pdf.py."""
    from fpdf import FPDF
    from scripts.make_demo_pdf import PAGES

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for text in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, text)
    path = tmp_path_factory.mktemp("demo") / "attention-summary.pdf"
    pdf.output(str(path))
    return path


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def test_chunking_is_page_aware(demo_pdf):
    from app import chunking

    pages = chunking.extract_pages(demo_pdf)
    assert len(pages) == 3
    assert "Encoder Stack" in pages[1]
    assert "BLEU" in pages[2]

    chunks = chunking.chunk_pages(pages)
    assert len(chunks) >= 2
    assert all(c.text.strip() for c in chunks)
    assert chunks[0].page_start == 1          # starts on page 1
    assert chunks[-1].page_end == 3           # last content lives on page 3
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_overlap_carries_context():
    from app import chunking, config

    config.CHUNK_SIZE = 120   # shrink so packing splits quickly
    config.CHUNK_OVERLAP = 40
    try:
        paras = ["A" * 80 + f" para{i}" for i in range(4)]
        chunks = chunking.chunk_pages(paras)
        assert len(chunks) >= 2
        # second chunk should start with carried tail of the first
        assert chunks[1].text.startswith(chunks[0].text[-40:])
    finally:
        config.CHUNK_SIZE = 900
        config.CHUNK_OVERLAP = 150


# ----------------------------------------------------------------------
# Vector store
# ----------------------------------------------------------------------
def test_store_roundtrip_and_dedup(demo_pdf):
    from app import chunking, store

    chunks, meta = chunking.ingest_pdf(demo_pdf)
    summary = store.ingest_document(demo_pdf, chunks)
    assert summary["num_chunks"] == len(chunks)

    hits = store.search("how many BLEU did the big model reach", k=2)
    assert hits, "expected at least one hit"
    assert "BLEU" in hits[0]["text"]
    assert hits[0]["metadata"]["source"].endswith(".pdf")

    # Re-uploading the same document must replace, not duplicate.
    store.ingest_document(demo_pdf, chunks)
    doc_id = summary["doc_id"]
    assert store.count_chunks(doc_id) == len(chunks)

    assert store.delete_document(doc_id) == len(chunks)


# ----------------------------------------------------------------------
# API end-to-end (generative when local Ollama is running; extractive otherwise)
# ----------------------------------------------------------------------
def test_api_full_flow(demo_pdf):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    # health
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    # llm_ready depends on a local Ollama daemon (CI and cold machines run
    # without one — the app then answers via the built-in extractive engine).
    assert isinstance(health["llm_ready"], bool)
    assert health["uploads_enabled"] is True
    assert health["deployment"] == "local"

    # upload
    with demo_pdf.open("rb") as fh:
        up = client.post(
            "/api/documents",
            files={"files": ("attention-summary.pdf", fh, "application/pdf")},
        )
    assert up.status_code == 200
    ingested = up.json()["ingested"][0]
    assert ingested["ok"] is True, ingested
    doc_id = ingested["doc_id"]

    # list
    docs = client.get("/api/documents").json()["documents"]
    match = [d for d in docs if d["doc_id"] == doc_id]
    assert match and match[0]["chunks"] == ingested["num_chunks"]

    # raw vector search
    found = client.post(
        "/api/search", json={"query": "scaled dot-product attention formula", "k": 2}
    ).json()
    assert found["results"], "search returned nothing"
    assert "softmax" in found["results"][0]["text"].lower()

    # RAG ask -> cited answer from the live local LLM
    asked = client.post(
        "/api/ask",
        json={"question": "How many GPUs were used to train the model?"},
    )
    assert asked.status_code == 200, asked.text
    data = asked.json()
    assert data["answer"].strip()
    assert len(data["sources"]) >= 1
    assert isinstance(data["citations"], list)
    for c in data["citations"]:
        assert set(c) == {"ref", "source", "pages", "score", "auto"}

    # scoped retrieval by doc_id still works
    scoped = client.post(
        "/api/search",
        json={"query": "encoder layers", "k": 3, "doc_id": doc_id},
    ).json()
    assert scoped["results"]

    # delete
    removed = client.delete(f"/api/documents/{doc_id}").json()
    assert removed["chunks_removed"] == ingested["num_chunks"]
