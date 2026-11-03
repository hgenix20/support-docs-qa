"""POST /ask: embed the question, retrieve closest chunks, answer with Bedrock, log the query."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from handlers.common import get_deps, parse_json_body, run
from rag.service import answer_question, parse_ask


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:  # noqa: ARG001 - Lambda signature
    return run(lambda: _ask(event))


def _ask(event: dict[str, Any]) -> dict[str, Any]:
    deps = get_deps()
    req = parse_ask(parse_json_body(event))
    result = answer_question(
        req,
        embedder=deps.embedder,
        generator=deps.generator,
        repo=deps.repo,
        now=datetime.now(UTC),
        query_id=str(uuid.uuid4()),
    )
    return {
        "query_id": result.query_id,
        "answer": result.answer,
        "sources": [
            {
                "id": i,
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.title,
                "chunk_index": hit.chunk.chunk_index,
                "score": round(hit.score, 4),
            }
            for i, hit in enumerate(result.hits)
        ],
    }
