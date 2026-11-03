"""Interfaces the service layer depends on. Adapters in rag/aws.py implement them.

Keeping these as Protocols means the business logic in rag/service.py runs in
unit tests with in-memory fakes and no AWS calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class CorpusTooLarge(Exception):
    """Raised by a repository when a full scan would exceed the configured cap."""


@dataclass(frozen=True)
class StoredChunk:
    doc_id: str
    title: str
    chunk_index: int
    text: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    question: str
    answer: str
    source_keys: list[str]
    created_at: datetime
    expires_at: int  # epoch seconds; DynamoDB TTL deletes the row after this


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class Repository(Protocol):
    def replace_document(self, doc_id: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        """Atomic-enough replace: old chunks are removed, then new ones written. Retry-safe."""
        ...

    def load_all(self, max_items: int) -> list[StoredChunk]:
        """Scan every chunk. Raises CorpusTooLarge if more than max_items exist."""
        ...

    def record_query(self, record: QueryRecord) -> None: ...
