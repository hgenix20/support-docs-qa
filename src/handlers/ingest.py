"""POST /documents: chunk, embed, and store one document (replaces any previous version)."""

from __future__ import annotations

from typing import Any

from handlers.common import get_deps, parse_json_body, run
from rag.service import ingest_document, parse_ingest


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:  # noqa: ARG001 - Lambda signature
    return run(lambda: _ingest(event))


def _ingest(event: dict[str, Any]) -> dict[str, Any]:
    deps = get_deps()
    req = parse_ingest(parse_json_body(event))
    count = ingest_document(req, embedder=deps.embedder, repo=deps.repo)
    return {"doc_id": req.doc_id, "chunks": count}
