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
            if len(para) >= MIN_PARA_LEN:
                paragraphs.append((page_no, para))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_pages: list[int] = []

    def flush() -> None:
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    page_start=min(buf_pages),
                    page_end=max(buf_pages),
                    index=len(chunks),
                )
            )

    for page_no, para in paragraphs:
        projected = sum(len(p) for p in buf) + len(para) + 2 * max(len(buf) - 1, 0)
        if buf and projected > config.CHUNK_SIZE:
            flush()
            # carry a tail from the previous chunk for overlap continuity
            tail = buf[-1][-config.CHUNK_OVERLAP:]
            last_page = buf_pages[-1]
            buf, buf_pages = ([tail], [last_page]) if tail.strip() else ([], [])
        buf.append(para)
        buf_pages.append(page_no)
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
