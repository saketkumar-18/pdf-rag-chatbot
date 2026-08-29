"""PDF parsing and page-aware semantic chunking.

Design notes
------------
* Every extracted chunk remembers which PDF page(s) it came from, so answers
  can cite "filename, p.4" instead of a meaningless blob id.
* Chunks are built from *paragraphs*: pages are split into blocks of text,
  adjacent paragraphs are packed greedily up to CHUNK_SIZE chars, and each new
  chunk starts with a small carried-over tail from the previous one so
  sentences cut at a boundary still appear fully inside some chunk (overlap).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config


@dataclass
class Chunk:
    text: str
    page_start: int  # 1-based inclusive
    page_end: int    # 1-based inclusive
    index: int       # position within the document


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
MIN_PARA_LEN = 20  # ignore page numbers / stray headers / footers


def _split_oversized(text: str, limit: int) -> list[str]:
    """Hard-split a paragraph longer than `limit` (e.g. a whole page with no
    blank lines, common with pdf.js extraction). Prefers sentence boundaries,
    falls back to word boundaries, then a hard cut. Resulting pieces are
    guaranteed to be <= limit."""
    parts: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < limit // 4:
            cut = limit
        else:
            cut += 1  # keep the delimiter with the left part
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        # Final safety: ensure the last piece is also within limit
        if len(text) > limit:
            # This shouldn't normally happen, but just in case
            parts.extend(_split_oversized(text, limit))
        else:
            parts.append(text)
    return parts


def _clean(text: str) -> str:
    """Normalize whitespace and drop hyphenation artifacts from line breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "docu-\nment" -> "document"
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return one cleaned text string per page (1-based ordering)."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a broken page must not kill ingestion
            raw = ""
        pages.append(_clean(raw))
    return pages


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Convert per-page texts into overlapping, page-aware chunks."""
    paragraphs: list[tuple[int, str]] = []
    for page_no, page_text in enumerate(pages, start=1):
        for para in _PARAGRAPH_SPLIT.split(page_text):
            para = para.strip()
            if len(para) < MIN_PARA_LEN:
                continue
            # A paragraph longer than a full chunk (whole page with no blank
            # lines) must be split or it can never fit the packing budget.
            if len(para) > config.CHUNK_SIZE:
                for piece in _split_oversized(para, config.CHUNK_SIZE):
                    if len(piece) >= MIN_PARA_LEN:
                        paragraphs.append((page_no, piece))
            else:
                paragraphs.append((page_no, para))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_pages: list[int] = []

    def flush() -> None:
        text = "\n\n".join(buf).strip()
        if text:
            # HARD CAP (safety net): if the assembled chunk still exceeds the
            # budget, split it. The carried overlap tail duplicates content the
            # previous chunk already holds, so it gets an allowance on top of
            # CHUNK_SIZE — this keeps the overlap contract intact while still
            # guaranteeing no pathological oversized chunks.
            limit = config.CHUNK_SIZE + config.CHUNK_OVERLAP
            if len(text) > limit:
                pieces = _split_oversized(text, limit)
                for piece in pieces:
                    chunks.append(
                        Chunk(
                            text=piece,
                            page_start=min(buf_pages),
                            page_end=max(buf_pages),
                            index=len(chunks),
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        text=text,
                        page_start=min(buf_pages),
                        page_end=max(buf_pages),
                        index=len(chunks),
                    )
                )

    for page_no, para in paragraphs:
        # Pre-calculate projected size INCLUDING the overlap tail that will be
        # carried forward from the previous chunk (if any). This ensures the
        # actual chunk content (tail + "\n\n" + new paragraphs) never exceeds
        # CHUNK_SIZE.
        tail = buf[-1][-config.CHUNK_OVERLAP:] if buf else ""
        tail_len = len(tail) + (2 if tail and buf else 0)  # "\n\n" separator
        projected = tail_len + sum(len(p) for p in buf) + len(para) + 2 * max(len(buf) - 1, 0)
        if buf and projected > config.CHUNK_SIZE:
            flush()
            # carry a tail from the previous chunk for overlap continuity
            tail = buf[-1][-config.CHUNK_OVERLAP:]
            last_page = buf_pages[-1]
            buf, buf_pages = ([tail], [last_page]) if tail.strip() else ([], [])
            # Check if overlap tail itself exceeds chunk size (shouldn't happen
            # but safe-guard)
            if buf and len(tail) > config.CHUNK_SIZE:
                flush()
                buf, buf_pages = [], []
        buf.append(para)
        buf_pages.append(page_no)
        # Hard guard: if a single paragraph exceeds CHUNK_SIZE after all
        # splitting attempts, flush immediately to prevent oversized chunks
        if len(buf) == 1 and len(para) > config.CHUNK_SIZE:
            flush()
            buf, buf_pages = [], []
    flush()

    # Renumber sequentially now that packing is final.
    for i, chunk in enumerate(chunks):
        chunk.index = i
    return chunks


def ingest_pdf(pdf_path: str | Path) -> tuple[list[Chunk], dict]:
    """Extract + chunk a PDF. Returns (chunks, document_metadata)."""
    reader = PdfReader(str(pdf_path))
    meta = {
        "num_pages": len(reader.pages),
        "title": (reader.metadata or {}).get("/Title") or Path(pdf_path).stem,
    }
    pages = extract_pages(pdf_path)
    return chunk_pages(pages), meta
