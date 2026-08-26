#!/usr/bin/env python3
"""Build the read-only flat-file index used by serverless deployments.

Usage:
    python scripts/build_index.py path/to/file.pdf [more.pdf ...]
    RAG_INDEX_OUT=data/index.json python scripts/build_index.py docs/*.pdf

Reads each PDF, chunks it page-aware, embeds chunks, and writes data/index.json:

    {"version": 1, "embedding_model": "...", "dim": 384,
     "chunks": [{"text", "vector", "metadata": {source, doc_id, page_start,
                 page_end, chunk_index}}, ...]}

Embedding source (same all-MiniLM-L6-v2 space either way):
    RAG_INDEX_EMBEDDINGS=local (default) -> fastembed ONNX, offline
    RAG_INDEX_EMBEDDINGS=hf              -> HF serverless feature-extraction
                                            API (needs HF_TOKEN env var)

The JSON backend normalizes vectors at load time; we store them unnormalized.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import chunking, config, store  # noqa: E402

_HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
_BATCH = 16


def _embed_hf(texts: list[str]) -> list[list[float]]:
    """Embed via HF serverless feature-extraction API (pooled sentence vecs)."""
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("RAG_INDEX_EMBEDDINGS=hf requires the HF_TOKEN env var.")
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        req = urllib.request.Request(
            _HF_EMBED_URL,
            data=json.dumps({"inputs": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                part = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"HF embedding HTTP {exc.code}: {exc.read()[:200]}") from exc
        if len(part) != len(batch):
            raise SystemExit(f"HF returned {len(part)} vectors for {len(batch)} inputs")
        vectors.extend(list(map(float, v)) for v in part)
        print(f"  embedded {min(i + _BATCH, len(texts))}/{len(texts)} chunks (hf)")
    return vectors


def _embed_local(texts: list[str]) -> list[list[float]]:
    from fastembed import TextEmbedding

    global _LOCAL_MODEL
    try:
        _LOCAL_MODEL
    except NameError:
        _LOCAL_MODEL = TextEmbedding(model_name=config.EMBEDDING_MODEL_SERVERLESS)
    return [list(map(float, v)) for v in _LOCAL_MODEL.embed(texts)]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        raise SystemExit(
            "\nError: pass at least one PDF. Example:\n"
            "  python scripts/build_index.py data/demo/attention-is-all-you-need-summary.pdf"
        )

    use_hf = os.getenv("RAG_INDEX_EMBEDDINGS", "local").strip().lower() == "hf"
    print(f"Embeddings: {'HF serverless API' if use_hf else 'fastembed (local ONNX)'}")

    out_path = Path(config.PREBUILT_INDEX)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse existing index so multiple runs ACCUMULATE documents instead of
    # overwriting (delete + rebuild = re-run build_index.py without args? no —
    # delete the file to start fresh).
    chunks: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            chunks = [
                c for c in existing.get("chunks", [])
                if isinstance(c, dict) and "vector" in c
            ]
            print(f"Loaded existing index: {len(chunks)} chunks")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not parse existing index ({exc}); starting fresh")

    total_new = 0
    for pdf in args:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"SKIP (missing): {pdf_path}")
            continue
        parsed, meta = chunking.ingest_pdf(pdf_path)
        if not parsed:
            print(f"SKIP (no text): {pdf_path.name}")
            continue
        doc_id = store.make_doc_id(pdf_path.name, pdf_path.stat().st_size)
        texts = [c.text for c in parsed]
        vectors = _embed_hf(texts) if use_hf else _embed_local(texts)
        for chunk_obj, vec in zip(parsed, vectors):
            chunks.append(
                {
                    "text": chunk_obj.text,
                    "vector": vec,
                    "metadata": {
                        "source": pdf_path.name,
                        "doc_id": doc_id,
                        "page_start": chunk_obj.page_start,
                        "page_end": chunk_obj.page_end,
                        "chunk_index": chunk_obj.index,
                    },
                }
            )
        total_new += len(parsed)
        print(f"Indexed {pdf_path.name}: {meta['num_pages']} pages -> {len(parsed)} chunks")

    payload = {
        "version": 1,
        "embedding_model": config.EMBEDDING_MODEL_SERVERLESS,
        "dim": len(chunks[0]["vector"]) if chunks else config.EMBEDDING_DIM,
        "num_documents": len({c["metadata"]["doc_id"] for c in chunks}),
        "chunks": chunks,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB): {total_new} new / {len(chunks)} total chunks")


if __name__ == "__main__":
    main()
