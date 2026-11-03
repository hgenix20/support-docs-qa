"""Use cases: ingest a document, answer a question. Pure logic over the ports.

Side effects (AWS calls, clock, UUIDs) enter through arguments, so every rule here
is testable without a network. HTTP status mapping lives in ApiError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from rag.chunking import chunk_text
from rag.ports import CorpusTooLarge, Embedder, Generator, QueryRecord, Repository, StoredChunk
from rag.prompt import Source, build_prompt
from rag.vectors import rank_top_k

# Bounds. Each is enforced at a boundary, so one request cannot exhaust time or memory.
MAX_TEXT_CHARS = 40_000
MAX_CHUNKS_PER_DOC = 40  # sequential embedding calls must fit the 29 s API Gateway limit
MAX_QUESTION_CHARS = 1_000
MAX_TOP_K = 8
DEFAULT_TOP_K = 4
MAX_TITLE_CHARS = 200
MAX_SCAN_CHUNKS = 2_000  # every ask scans the corpus; this caps the per-query read cost
CHUNK_CHARS = 800
CHUNK_OVERLAP = 100
QUERY_TTL_DAYS = 30

DOC_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class ApiError(Exception):
    """An error with an HTTP status that is safe to show the caller."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class IngestRequest:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class AskRequest:
    question: str
    top_k: int


@dataclass(frozen=True)
class Hit:
    chunk: StoredChunk
    score: float


@dataclass(frozen=True)
class AskResult:
    query_id: str
    answer: str
    hits: list[Hit]


def parse_ingest(body: Any) -> IngestRequest:
    if not isinstance(body, dict):
        raise ApiError(400, "body must be a JSON object")
    doc_id = body.get("doc_id")
    if not isinstance(doc_id, str) or not DOC_ID_PATTERN.fullmatch(doc_id):
        raise ApiError(400, "doc_id must be 1-64 chars of lowercase letters, digits, hyphens")
    title = body.get("title")
    if not isinstance(title, str) or not 0 < len(title.strip()) <= MAX_TITLE_CHARS:
        raise ApiError(400, f"title must be a non-empty string up to {MAX_TITLE_CHARS} chars")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, "text must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        raise ApiError(413, f"text exceeds {MAX_TEXT_CHARS} chars")
    return IngestRequest(doc_id=doc_id, title=title.strip(), text=text)


def parse_ask(body: Any) -> AskRequest:
    if not isinstance(body, dict):
        raise ApiError(400, "body must be a JSON object")
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ApiError(400, "question must be a non-empty string")
    if len(question) > MAX_QUESTION_CHARS:
        raise ApiError(400, f"question exceeds {MAX_QUESTION_CHARS} chars")
    top_k = body.get("top_k", DEFAULT_TOP_K)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ApiError(400, f"top_k must be an integer from 1 to {MAX_TOP_K}")
    return AskRequest(question=question.strip(), top_k=top_k)


def ingest_document(req: IngestRequest, *, embedder: Embedder, repo: Repository) -> int:
    """Chunk, embed, and store one document. Returns the number of chunks written."""
    pieces = chunk_text(req.text, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP)
    if not pieces:
        raise ApiError(400, "text is empty after trimming")
    if len(pieces) > MAX_CHUNKS_PER_DOC:
        raise ApiError(
            413, f"document would produce {len(pieces)} chunks; limit is {MAX_CHUNKS_PER_DOC}"
        )
    stored = [
        StoredChunk(
            doc_id=req.doc_id,
            title=req.title,
            chunk_index=piece.index,
            text=piece.text,
            vector=tuple(embedder.embed(piece.text)),
        )
        for piece in pieces
    ]
    repo.replace_document(req.doc_id, req.title, stored)
    return len(stored)


def answer_question(
    req: AskRequest,
    *,
    embedder: Embedder,
    generator: Generator,
    repo: Repository,
    now: datetime,
    query_id: str,
) -> AskResult:
    try:
        corpus = repo.load_all(max_items=MAX_SCAN_CHUNKS)
    except CorpusTooLarge as exc:
        raise ApiError(503, f"corpus exceeds the {MAX_SCAN_CHUNKS}-chunk retrieval limit") from exc
    if not corpus:
        raise ApiError(503, "no documents have been ingested yet")

    query_vector = embedder.embed(req.question)
    ranked = rank_top_k(query_vector, [(c, c.vector) for c in corpus], req.top_k)
    hits = [Hit(chunk=chunk, score=score) for chunk, score in ranked]

    system, user = build_prompt(
        req.question,
        [
            Source(
                doc_id=h.chunk.doc_id,
                title=h.chunk.title,
                chunk_index=h.chunk.chunk_index,
                text=h.chunk.text,
            )
            for h in hits
        ],
    )
    answer = generator.generate(system, user)

    repo.record_query(
        QueryRecord(
            query_id=query_id,
            question=req.question,
            answer=answer,
            source_keys=[f"{h.chunk.doc_id}#{h.chunk.chunk_index}" for h in hits],
            created_at=now,
            expires_at=int((now + timedelta(days=QUERY_TTL_DAYS)).timestamp()),
        )
    )
    return AskResult(query_id=query_id, answer=answer, hits=hits)
