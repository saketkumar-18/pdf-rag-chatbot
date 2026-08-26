"""Extractive answer engine — zero-dependency fallback when no LLM is available.

Splits the question into content words, ranks the sentences of each retrieved
chunk by keyword overlap, and returns the best 2-4 sentences as a cited
answer. Not as fluent as an LLM, but grounded, fast, free, and always
available (important for public demos where you don't want hard failures).
"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "will", "would", "should", "shall",
    "of", "in", "on", "at", "to", "for", "with", "about", "from", "by", "as",
    "and", "or", "but", "not", "no", "it", "its", "this", "that", "these",
    "those", "there", "their", "they", "them", "he", "she", "we", "you",
    "i", "me", "my", "your", "our", "us", "if", "then", "than", "so", "such",
    "into", "over", "under", "between", "also", "more", "most", "some", "any",
    "tell", "say", "give", "explain", "describe", "using", "used", "use",
}

# A question this generic can't be answered extractively; say so instead.
_UNANSWERABLE_HINTS = {"summary", "summarize", "summarise", "tldr"}


def _content_words(text: str) -> list[str]:
    return [
        w for w in _WORD.findall(text.lower())
        if len(w) > 1 and w not in _STOPWORDS
    ]


def extractive_answer(question: str, hits: list[dict]) -> str:
    """Best-matching sentences from the retrieved chunks, with [n] markers."""
    q_words = set(_content_words(question))
    if not q_words or any(w in question.lower() for w in _UNANSWERABLE_HINTS):
        return (
            "I can retrieve passages, but I can't summarize without a generative "
            "model. Ask about a specific fact from the documents (e.g. a name, "
            "number, or method), or ask the site owner to add an LLM API key."
        )

    scored_sentences: list[tuple[float, int, str]] = []
    for block_no, hit in enumerate(hits, start=1):
        text = hit["text"]
        for sent in _SENT_SPLIT.split(text):
            words = _content_words(sent)
            if not words:
                continue
            overlap = sum(1 for w in words if w in q_words)
            if overlap == 0:
                continue
            # density matters: short sentences with high overlap win
            score = overlap / (1 + len(words) ** 0.5)
            scored_sentences.append((score, block_no, sent.strip()))

    if not scored_sentences:
        return (
            "I couldn't find anything relevant to that in the indexed "
            "documents. Try different wording, or ask about something the "
            "documents actually cover."
        )

    scored_sentences.sort(key=lambda t: t[0], reverse=True)

    # Take top sentences but keep them in document order for readability.
    picked = scored_sentences[:3]
    picked.sort(key=lambda t: (t[1], -t[0]))

    lines = []
    used_blocks: list[int] = []
    for _, block_no, sent in picked:
        marker = f"[{block_no}]"
        if not used_blocks or used_blocks[-1] != block_no:
            lines.append(f"{sent} {marker}")
            used_blocks.append(block_no)
        else:
            # merge continuation sentence into previous line
            lines[-1] = f"{lines[-1]} {sent}"

    note = "_Answered by built-in retrieval mode (no LLM key configured)._"
    return "\n\n".join(lines) + f"\n\n{note}"
