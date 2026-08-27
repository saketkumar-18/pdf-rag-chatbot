"""FastAPI application: upload, search, ask, manage documents."""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import chunking, config, llm, prompts, store

app = FastAPI(
    title="PDF RAG Chatbot",
    description="Upload PDFs, ask questions, get answers cited to page numbers.",
    version="2.0.0",
)

# --------------------------------------------------------------------------
# CORS: same-origin always works; extras via RAG_CORS_ORIGINS env var.
# Wildcard only as an explicit opt-in.
# --------------------------------------------------------------------------
_DEFAULT_CORS = ["http://localhost:8000", "http://127.0.0.1:8000"]
if config.CORS_ORIGINS == ["*"]:
    _allow_origins = ["*"]
elif config.CORS_ORIGINS:
    _allow_origins = _DEFAULT_CORS + config.CORS_ORIGINS
else:
    _allow_origins = _DEFAULT_CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=5, ge=1, le=20)
    doc_id: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    doc_id: str | None = None
    agent: bool | None = None  # None = server decides (RAG_AGENT_MODE)
    mode: str | None = None    # "classic" | "agent" | "research" (overrides `agent`)
    # Browser-indexed documents (client-side retrieval): the client sends its
    # own top-k chunks; the server only generates the cited answer.
    contexts: list[dict] | None = None


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)


class Citation(BaseModel):
    ref: int
    source: str
    pages: str
    score: float
    auto: bool = False


# --------------------------------------------------------------------------
# Rate limiting (per-IP sliding window, in-memory)
# --------------------------------------------------------------------------
_buckets: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request, kind: str) -> None:
    if not config.RATE_LIMIT_ENABLED:
        return
    limit = {
        "ask": config.RATE_LIMIT_ASK,
        "search": config.RATE_LIMIT_SEARCH,
        "upload": config.RATE_LIMIT_UPLOAD,
        "embed": config.RATE_LIMIT_EMBED,
    }[kind]
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _buckets[ip + ":" + kind]
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(429, "Rate limit exceeded — try again in a minute.")
    bucket.append(now)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _validate_and_save(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Only PDF files are accepted (got '{suffix}').")
    safe_name = Path(upload.filename).name
    dest = config.UPLOADS_DIR / f"{int(time.time() * 1000)}_{safe_name}"
    size = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB limit.")
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file.")
    return dest


def _parse_citations(answer: str, hits: list[dict]) -> tuple[str, list[Citation]]:
    refs = []
    for match in re.findall(r"\[(\d+)\]", answer):
        n = int(match)
        if 1 <= n <= len(hits) and n not in refs:
            refs.append(n)

    def make_citation(n: int, auto: bool) -> Citation:
        meta = hits[n - 1]["metadata"]
        pages = (
            f"p.{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"pp.{meta['page_start']}-{meta['page_end']}"
        )
        return Citation(
            ref=n,
            source=meta["source"],
            pages=pages,
            score=hits[n - 1]["score"],
            auto=auto,
        )

    if refs:
        return answer, [make_citation(n, auto=False) for n in refs]

    SIMILARITY_FLOOR = 0.25
    auto_hits = [
        i + 1 for i, h in enumerate(hits[:3]) if h["score"] >= SIMILARITY_FLOOR
    ] or [1]
    return answer, [make_citation(n, auto=True) for n in auto_hits]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts server-side via the HF feature-extraction proxy.

    Used by /api/embed so browser-indexed uploads never see the HF token.
    Raises HTTPException on any failure (the UI surfaces the message).
    """
    import json as _json
    import urllib.error
    import urllib.request

    if not config.HF_TOKEN:
        raise HTTPException(
            503, "Embedding proxy is not configured on this deployment (no HF_TOKEN)."
        )
    req = urllib.request.Request(
        config.HF_EMBED_URL,
        data=_json.dumps({"inputs": texts}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.HF_TOKEN}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            vectors = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HTTPException(502, f"Embedding service error (HTTP {exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(502, f"Embedding service unreachable ({exc.reason}).") from exc
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise HTTPException(502, "Embedding service returned an unexpected payload.")
    return [[float(x) for x in v] for v in vectors]


def _client_hits(contexts: list[dict]) -> list[dict]:
    """Validate + normalize client-provided chunks into the standard hit shape."""
    hits: list[dict] = []
    for i, c in enumerate(contexts[:10]):
        text = str(c.get("text", ""))[:6000]
        if not text.strip():
            continue
        try:
            page_start = max(1, int(c.get("page_start", 1)))
            page_end = max(page_start, int(c.get("page_end", page_start)))
        except (TypeError, ValueError):
            page_start = page_end = 1
        hits.append(
            {
                "id": f"client_c{i}",
                "text": text,
                "metadata": {
                    "source": str(c.get("source", "your document"))[:200],
                    "doc_id": "client",
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": i,
                },
                "score": float(c.get("score", 0.0)),
            }
        )
    return hits


def _agent_response(result: dict, t0: float) -> dict:
    """Shape a LangGraph agent result into the standard /api/ask contract."""
    hits = result["cited_hits"] or result["all_hits"]
    answer, citations = _parse_citations(result["answer"], hits)
    total_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "answer": answer,
        "citations": [c.model_dump() for c in citations],
        "sources": [
            {
                "text": h["text"],
                "source": h["metadata"]["source"],
                "page_start": h["metadata"]["page_start"],
                "page_end": h["metadata"]["page_end"],
                "score": h["score"],
            }
            for h in hits
        ],
        "timing": {
            "retrieval_ms": sum(s["ms"] for s in result["trace"] if s["node"] == "retrieve"),
            "generation_ms": sum(s["ms"] for s in result["trace"] if s["node"] == "generate"),
            "best_similarity": hits[0]["score"] if hits else 0.0,
        },
        "agent": {
            "enabled": True,
            "mode": result.get("mode", "agent"),
            "trace": result["trace"],
            "rewrites": result["rewrites"],
            "retries": result["retries"],
            "verdict": result["verdict"],
            "subqueries": result.get("subqueries", []),
            "total_ms": total_ms,
        },
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def _health_llm_label() -> str:
    backend = config.effective_llm_backend()
    if backend == "ollama":
        return config.LLM_MODEL
    if backend == "extractive":
        return "built-in retrieval"
    # openai-compatible: mirror llm._openai_generate routing exactly
    if config.OPENAI_API_KEY:
        return config.OPENAI_LLM_MODEL
    return f"{config.HF_LLM_MODEL} (hf)"


def _health_embed_label() -> str:
    return (
        config.EMBEDDING_MODEL_SERVERLESS
        if store._resolve_backend() == "json"
        else config.EMBEDDING_MODEL
    )


@app.get("/api/health")
def health():
    from . import agent

    try:
        chunks = store.count_chunks()
    except FileNotFoundError:
        # serverless: index file not bundled -> report degraded, not crash
        return {
            "status": "degraded",
            "llm_ready": llm.llm_alive(),
            "llm_backend": config.effective_llm_backend(),
            "llm_model": _health_llm_label(),
            "embedding_model": _health_embed_label(),
            "chunks_indexed": 0,
            "deployment": config.DEPLOYMENT,
            "uploads_enabled": False,
            "browser_uploads": bool(config.HF_TOKEN),
            "agent": {
                "mode": config.AGENT_MODE,
                "available": agent.available(),
                "langgraph": agent.LANGGRAPH_VERSION,
            },
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Vector store unavailable: {exc}") from exc
    return {
        "status": "ok",
        "llm_ready": llm.llm_alive(),
        "llm_backend": config.effective_llm_backend(),
        "llm_model": _health_llm_label(),
        "embedding_model": _health_embed_label(),
        "chunks_indexed": chunks,
        "deployment": config.DEPLOYMENT,
        "uploads_enabled": not config.SERVERLESS,
        "browser_uploads": bool(config.HF_TOKEN),
        "agent": {
            "mode": config.AGENT_MODE,
            "available": agent.available(),
            "langgraph": agent.LANGGRAPH_VERSION,
        },
    }


@app.post("/api/documents")
async def upload_documents(
    request: Request, files: list[UploadFile] | None = File(None)
):
    if config.SERVERLESS:
        raise HTTPException(
            403,
            "This is a read-only demo deployment — uploads are disabled here. "
            "Run the project locally for full upload support (see README).",
        )
    if not files:
        raise HTTPException(422, "No files were uploaded (multipart field 'files').")
    _rate_limit(request, "upload")
    results = []
    for upload in files:
        saved_path = _validate_and_save(upload)
        try:
            chunks, meta = chunking.ingest_pdf(saved_path)
            if not chunks:
                results.append(
                    {
                        "filename": upload.filename,
                        "ok": False,
                        "error": "No extractable text found (scanned/image PDF?). OCR is not supported yet.",
                    }
                )
                continue
            summary = store.ingest_document(saved_path, chunks)
            results.append(
                {
                    "filename": upload.filename,
                    "ok": True,
                    "doc_id": summary["doc_id"],
                    "num_pages": meta["num_pages"],
                    "num_chunks": summary["num_chunks"],
                }
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            results.append({"filename": upload.filename, "ok": False, "error": str(exc)})
    return {"ingested": results}


@app.get("/api/documents")
def list_documents():
    docs = store.list_documents_backend_agnostic()
    return {"documents": docs}


@app.delete("/api/documents/{doc_id}")
def delete_document(request: Request, doc_id: str):
    if config.SERVERLESS:
        raise HTTPException(403, "Read-only deployment — deletes are disabled.")
    _rate_limit(request, "upload")
    removed = store.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(404, f"No chunks found for doc_id '{doc_id}'.")
    return {"deleted": doc_id, "chunks_removed": removed}


@app.post("/api/search")
def search(request: Request, req: SearchRequest):
    _rate_limit(request, "search")
    where = {"doc_id": {"$eq": req.doc_id}} if req.doc_id else None
    hits = store.search(req.query, k=req.k, where=where)
    return {
        "query": req.query,
        "results": [
            {
                "text": h["text"],
                "source": h["metadata"]["source"],
                "page_start": h["metadata"]["page_start"],
                "page_end": h["metadata"]["page_end"],
                "score": h["score"],
            }
            for h in hits
        ],
    }


@app.post("/api/embed")
def embed(request: Request, req: EmbedRequest):
    """Embedding proxy for browser-side document indexing.

    The UI parses/chunks PDFs locally, sends chunk texts here in batches,
    and stores the returned vectors in the browser (IndexedDB). The HF token
    never leaves the server.
    """
    _rate_limit(request, "embed")
    texts = [t.strip() for t in req.texts]
    if any(len(t) > 4000 for t in texts):
        raise HTTPException(422, "Each text must be at most 4000 characters.")
    if not all(texts):
        raise HTTPException(422, "Empty texts are not allowed.")
    vectors = _embed_texts(texts)
    return {"vectors": vectors, "dim": len(vectors[0]) if vectors else 0}


@app.post("/api/ask")
def ask(request: Request, req: AskRequest):
    _rate_limit(request, "ask")
    t0 = time.perf_counter()

    # ---- Client-provided contexts (browser-indexed documents) ------------
    # The client already retrieved its own top-k chunks; we only generate.
    if req.contexts:
        hits = _client_hits(req.contexts)
        if not hits:
            raise HTTPException(422, "No usable contexts were provided.")
        retrieval_ms = 0
        t1 = time.perf_counter()
        backend = config.effective_llm_backend()
        if backend == "extractive":
            from . import extractive

            answer = extractive.extractive_answer(req.question, hits)
        else:
            prompt = prompts.build_answer_prompt(req.question, hits)
            try:
                answer = llm.generate(prompt, system=prompts.SYSTEM_PROMPT)
            except llm.LLMError as exc:
                raise HTTPException(503, str(exc)) from exc
        generation_ms = int((time.perf_counter() - t1) * 1000)
        answer, citations = _parse_citations(answer, hits)
        return {
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "sources": [
                {
                    "text": h["text"],
                    "source": h["metadata"]["source"],
                    "page_start": h["metadata"]["page_start"],
                    "page_end": h["metadata"]["page_end"],
                    "score": h["score"],
                }
                for h in hits
            ],
            "timing": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "best_similarity": hits[0]["score"],
            },
            "client_indexed": True,
        }

    # ---- Agentic paths (LangGraph) ---------------------------------------
    from . import agent

    # Resolve which graph to run: explicit `mode` wins, then legacy `agent`
    # bool, then the server default (RAG_AGENT_MODE).
    graph_mode: str | None = None
    if req.mode in {"agent", "research"}:
        graph_mode = req.mode
    elif req.mode == "classic":
        graph_mode = None
    elif agent.resolve_mode(req.agent):
        graph_mode = "agent"

    if graph_mode:
        try:
            result = agent.run_agent(
                req.question, top_k=req.top_k, doc_id=req.doc_id, mode=graph_mode
            )
        except agent.AgentUnavailable:
            result = None  # fall through to the classic path below
        except llm.LLMError as exc:
            raise HTTPException(503, str(exc)) from exc
        except FileNotFoundError:
            raise HTTPException(
                503,
                "The knowledge index is not available on this deployment yet. "
                "The site owner needs to bundle data/index.json (see README).",
            )
        if result is not None:
            return _agent_response(result, t0)

    # ---- Classic single-shot path ----------------------------------------
    where = {"doc_id": {"$eq": req.doc_id}} if req.doc_id else None
    try:
        hits = store.search(req.question, k=req.top_k, where=where)
    except FileNotFoundError:
        raise HTTPException(
            503,
            "The knowledge index is not available on this deployment yet. "
            "The site owner needs to bundle data/index.json (see README).",
        )

    if not hits:
        return {
            "answer": "The document index is empty"
            + ("." if config.SERVERLESS else " — upload a PDF first."),
            "citations": [],
            "sources": [],
        }

    best_similarity = hits[0]["score"]

    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    t1 = time.perf_counter()

    backend = config.effective_llm_backend()
    if backend == "extractive":
        # No LLM configured: built-in retrieval answering (always available).
        from . import extractive

        answer = extractive.extractive_answer(req.question, hits)
        generation_ms = int((time.perf_counter() - t1) * 1000)
        answer, citations = _parse_citations(answer, hits)
        return {
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "sources": [
                {
                    "text": h["text"],
                    "source": h["metadata"]["source"],
                    "page_start": h["metadata"]["page_start"],
                    "page_end": h["metadata"]["page_end"],
                    "score": h["score"],
                }
                for h in hits
            ],
            "timing": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "best_similarity": best_similarity,
            },
        }

    prompt = prompts.build_answer_prompt(req.question, hits)
    try:
        answer = llm.generate(prompt, system=prompts.SYSTEM_PROMPT)
    except llm.LLMError as exc:
        raise HTTPException(503, str(exc)) from exc
    generation_ms = int((time.perf_counter() - t1) * 1000)

    answer, citations = _parse_citations(answer, hits)

    return {
        "answer": answer,
        "citations": [c.model_dump() for c in citations],
        "sources": [
            {
                "text": h["text"],
                "source": h["metadata"]["source"],
                "page_start": h["metadata"]["page_start"],
                "page_end": h["metadata"]["page_end"],
                "score": h["score"],
            }
            for h in hits
        ],
        "timing": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "best_similarity": best_similarity,
        },
    }


# --------------------------------------------------------------------------
# Security headers + static UI (mounted last so /api/* wins)
# --------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(config.BASE_DIR / "static" / "favicon.svg", media_type="image/svg+xml")


app.mount(
    "/",
    StaticFiles(directory=str(config.BASE_DIR / "static"), html=True),
    name="static",
)
