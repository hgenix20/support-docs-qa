"""Split document text into overlapping windows sized for embedding.

Contract: returns chunks with non-empty text, each at most ``max_chars`` long,
indexed 0..n-1 in document order. Consecutive chunks share up to ``overlap``
characters so a fact that straddles a boundary still appears whole in one chunk.
Raises ValueError on invalid parameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    n = len(cleaned)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # Prefer breaking on a space in the back half of the window so words stay whole.
            cut = cleaned.rfind(" ", start + max_chars // 2, end)
            if cut > start:
                end = cut
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(Chunk(index=len(chunks), text=piece))
        if end >= n:
            break
        # Guarantee forward progress even when overlap is large relative to the window.
        start = max(end - overlap, start + 1)
    return chunks
