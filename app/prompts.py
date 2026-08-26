"""Prompt construction for grounded, cited answers.

The contract we impose on the model:
1. Answer ONLY from the numbered context blocks.
2. Cite every claim inline as [1], [2] ... matching the block number.
3. If the context does not contain the answer, say so explicitly.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a precise document QA assistant. You answer questions \
strictly from the numbered context blocks provided by the user.

Rules:
1. Use ONLY facts present in the context blocks. Never invent information.
2. Cite sources inline after each claim using the block number in square brackets, e.g. [1] or [2][3].
3. If multiple blocks support one sentence, cite them together like [1][2].
4. If the context does not contain enough information to answer, reply exactly: \
"I don't have enough information in the uploaded documents to answer that." \
and optionally suggest what the user could upload instead.
5. Be concise: 1-4 short paragraphs unless asked for detail.

Example of the required style:
Question: What dataset was used?
Answer: The model was trained on the WMT 2014 English-German corpus with 4.5 million
sentence pairs [1], using a shared vocabulary of about 37000 tokens [1]."""

ANSWER_TEMPLATE = """Context blocks from the user's uploaded documents:

{context}

Question: {question}

Answer (cite blocks as [1], [2], ...):"""


def build_context_block(hits: list[dict]) -> str:
    """Render retrieved hits as numbered, cited context blocks."""
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        pages = (
            f"p.{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"pp.{meta['page_start']}-{meta['page_end']}"
        )
        lines.append(f"[{i}] Source: {meta['source']} ({pages})\n{hit['text']}")
    return "\n\n".join(lines)


def build_answer_prompt(question: str, hits: list[dict]) -> str:
    return ANSWER_TEMPLATE.format(
        context=build_context_block(hits), question=question.strip()
    )
