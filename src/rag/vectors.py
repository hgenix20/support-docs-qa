"""Cosine similarity and top-k ranking over in-memory vectors.

Contract: ``rank_top_k`` returns at most k (item, score) pairs, highest score first.
Vectors of mismatched length raise ValueError instead of scoring silently.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_top_k[T](
    query: Sequence[float],
    candidates: Iterable[tuple[T, Sequence[float]]],
    k: int,
) -> list[tuple[T, float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    scored = ((item, cosine(query, vec)) for item, vec in candidates)
    return heapq.nlargest(k, scored, key=lambda pair: pair[1])
