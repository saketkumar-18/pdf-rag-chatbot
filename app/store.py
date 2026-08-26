"""Embedding + vector-store layer with two interchangeable backends.

Backends
--------
chroma (default, RAG_STORE_BACKEND=chroma)
    PersistentClient + sentence-transformers embeddings. Writable: supports
    upload/delete at runtime. Used for local/Docker deployments.

json (RAG_STORE_BACKEND=json)
    Zero-dependency flat-file index (data/index.json) built ahead of time by
    scripts/build_index.py. Read-only at runtime: perfect for serverless
    platforms (Vercel) where the filesystem is immutable and heavyweight
    deps (torch/chromadb) would blow the bundle limit. Embeddings via
    fastembed (ONNX, no torch), exact cosine similarity in numpy-free stdlib
    Python over the bundled chunk set (fine for <= ~20k chunks).

Chunk metadata schema (both backends):
    source      file name as uploaded          -> shown in citations
    doc_id      stable id of the document      -> used for delete/filter
    page_start  first page of the chunk (1-based)
    page_end    last page of the chunk (1-based)
    chunk_index position within the document
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from pathlib import Path

from . import config

# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
def _resolve_backend() -> str:
    backend = getattr(config, "STORE_BACKEND", None)
    if backend:
        return backend
    # serverless -> bundled flat-file index; local -> chromadb
    return "json" if config.SERVERLESS else "chroma"


# Backwards-compatible alias (health endpoint calls this).
resolve_backend = _resolve_backend


def _get_embed_fn():
    """Return an embedding function according to the effective backend."""
    if config.effective_embedding_backend() == "fastembed":
        from fastembed import TextEmbedding

        model = config.EMBEDDING_MODEL_SERVERLESS

        class _FastEmbedFn:
            _model = None

            def __call__(self, texts):
                if _FastEmbedFn._model is None:
                    try:
                        # bundled cache -> never hit the network on cold start
                        _FastEmbedFn._model = TextEmbedding(
                            model_name=model,
                            cache_dir=config.CACHE_DIR,
                            local_files_only=True,
                        )
                    except TypeError:  # older fastembed w/o the kwarg
                        _FastEmbedFn._model = TextEmbedding(
                            model_name=model,
                            cache_dir=config.CACHE_DIR,
                        )
                return [list(map(float, v)) for v in _FastEmbedFn._model.embed(texts)]

        return _FastEmbedFn()

    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL,
        device="cpu",
    )


# --------------------------------------------------------------------------
# Chroma backend (local / Docker)
# --------------------------------------------------------------------------
_client = None
_embed_fn = None


def get_client():
    global _client
    if _client is None:
        import chromadb

        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection(name: str = "documents"):
    return get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embed_fn_singleton(),
    )


def _embed_fn_singleton():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = _get_embed_fn()
    return _embed_fn


# --------------------------------------------------------------------------
# JSON backend (serverless)
# --------------------------------------------------------------------------
_json_lock = threading.Lock()
_json_cache: dict | None = None  # {"version", "embedding_model", "dim", "chunks":[...]}


class ReadOnlyStoreError(RuntimeError):
    """Raised when a write op hits the read-only JSON backend."""


def _load_index() -> dict:
    global _json_cache
    if _json_cache is None:
        with _json_lock:
            if _json_cache is None:
                raw = config.PREBUILT_INDEX.read_text(encoding="utf-8")
                data = json.loads(raw)
                # normalize vectors to unit length once, at load time
                for ch in data.get("chunks", []):
                    v = ch["vector"]
                    norm = math.sqrt(sum(x * x for x in v)) or 1.0
                    ch["vector"] = [x / norm for x in v]
                _json_cache = data
    return _json_cache


def _embed_query(query: str) -> list[float]:
    fn = _embed_fn_singleton()
    vec = fn([query])[0]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _json_search(query: str, k: int, where: dict | None) -> list[dict]:
    data = _load_index()
    chunks = data["chunks"]
    if not chunks:
        return []
    qv = _embed_query(query)

    doc_filter = None
    if where:
        cond = where.get("doc_id", {})
        if isinstance(cond, dict):
            doc_filter = cond.get("$eq")

    scored = []
    for idx, ch in enumerate(chunks):
        if doc_filter and ch["metadata"].get("doc_id") != doc_filter:
            continue
        vec = ch["vector"]
        dot = sum(a * b for a, b in zip(qv, vec))
        scored.append((dot, idx, ch))
    scored.sort(key=lambda t: t[0], reverse=True)

    hits = []
    for score, _, ch in scored[:k]:
        hits.append(
            {
                "id": f"{ch['metadata']['doc_id']}_c{ch['metadata']['chunk_index']}",
                "text": ch["text"],
                "metadata": ch["metadata"],
                "distance": round(1.0 - score, 4),
                "score": round(score, 4),
            }
        )
    return hits


# --------------------------------------------------------------------------
# Public API (backend-agnostic, same shapes as before)
# --------------------------------------------------------------------------
def make_doc_id(filename: str, size: int) -> str:
    """Stable document id derived from name+size (re-upload = same id)."""
    digest = hashlib.sha1(f"{filename}:{size}".encode()).hexdigest()[:12]
    return f"doc_{digest}"


def ingest_document(pdf_path: Path, chunks) -> dict:
    """Embed chunks and store them (Chroma backend only)."""
    if _resolve_backend() != "chroma":
        raise ReadOnlyStoreError(
            "Uploads are disabled on this deployment (read-only index)."
        )
    doc_id = make_doc_id(pdf_path.name, pdf_path.stat().st_size)

    collection = get_collection()
    collection.delete(where={"doc_id": {"$eq": doc_id}})

    if not chunks:
        return {"doc_id": doc_id, "num_chunks": 0}

    ids = [f"{doc_id}_c{c.index}_{uuid.uuid4().hex[:8]}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source": pdf_path.name,
            "doc_id": doc_id,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "chunk_index": c.index,
        }
        for c in chunks
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return {"doc_id": doc_id, "num_chunks": len(ids)}


def search(query: str, k: int | None = None, where: dict | None = None) -> list[dict]:
    """Vector similarity search. Returns ranked hits with metadata + distance."""
    k = min(k or config.TOP_K, 50)

    if _resolve_backend() == "json":
        return _json_search(query, k, where)

    collection = get_collection()
    n = collection.count()
    result = collection.query(
        query_texts=[query],
        n_results=min(k, n or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for i in range(len(result["ids"][0])):
        distance = result["distances"][0][i]
        hits.append(
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": round(float(distance), 4),
                "score": round(1.0 - float(distance), 4),
            }
        )
    return hits


def count_chunks(doc_id: str | None = None) -> int:
    if _resolve_backend() == "json":
        data = _load_index()
        chunks = data["chunks"]
        if doc_id is None:
            return len(chunks)
        return sum(1 for c in chunks if c["metadata"].get("doc_id") == doc_id)

    collection = get_collection()
    if doc_id is None:
        return collection.count()
    got = collection.get(where={"doc_id": {"$eq": doc_id}})
    return len(got["ids"])


def list_documents_backend_agnostic() -> list[dict]:
    """Documents summary for both backends."""
    if _resolve_backend() == "json":
        docs: dict[str, dict] = {}
        for ch in _load_index()["chunks"]:
            meta = ch["metadata"]
            entry = docs.setdefault(
                meta["doc_id"],
                {
                    "doc_id": meta["doc_id"],
                    "source": meta["source"],
                    "chunks": 0,
                    "last_page": 0,
                },
            )
            entry["chunks"] += 1
            entry["last_page"] = max(entry["last_page"], meta["page_end"])
        return sorted(docs.values(), key=lambda d: d["source"])

    collection = get_collection()
    if collection.count() == 0:
        return []
    got = collection.get(include=["metadatas"])
    docs = {}
    for meta in got["metadatas"]:
        entry = docs.setdefault(
            meta["doc_id"],
            {
                "doc_id": meta["doc_id"],
                "source": meta["source"],
                "chunks": 0,
                "last_page": 0,
            },
        )
        entry["chunks"] += 1
        entry["last_page"] = max(entry["last_page"], meta["page_end"])
    return sorted(docs.values(), key=lambda d: d["source"])


def delete_document(doc_id: str) -> int:
    if _resolve_backend() != "chroma":
        raise ReadOnlyStoreError(
            "Deletes are disabled on this deployment (read-only index)."
        )
    collection = get_collection()
    before = collection.count()
    collection.delete(where={"doc_id": {"$eq": doc_id}})
    return before - collection.count()


def reset_all() -> int:
    if _resolve_backend() != "chroma":
        raise ReadOnlyStoreError("Reset is disabled on read-only deployments.")
    collection = get_collection()
    n = collection.count()
    get_client().delete_collection(collection.name)
    return n
