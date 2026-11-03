"""Lambda plumbing shared by both handlers: body parsing, error-to-HTTP mapping, cached clients.

Contract: ``run`` always returns a valid API Gateway proxy response. Known errors
(ApiError) return their status and message. Anything else returns a generic 500
and logs the full traceback, so internal details never reach the caller.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import boto3

from rag.aws import BedrockEmbedder, BedrockGenerator, DynamoRepository, Settings
from rag.ports import Embedder, Generator, Repository
from rag.service import ApiError

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 256_000  # bounds JSON parsing; the largest legal text is 40k chars


@dataclass(frozen=True)
class Deps:
    embedder: Embedder
    generator: Generator
    repo: Repository


_deps: Deps | None = None


def get_deps() -> Deps:
    """Build clients on first use and reuse them across warm invocations."""
    global _deps
    if _deps is None:
        settings = Settings.from_env(_environ())
        bedrock = boto3.client("bedrock-runtime")
        dynamodb = boto3.resource("dynamodb")
        _deps = Deps(
            embedder=BedrockEmbedder(bedrock, settings.embed_model_id),
            generator=BedrockGenerator(bedrock, settings.generation_model_id),
            repo=DynamoRepository(dynamodb, settings),
        )
    return _deps


def parse_json_body(event: dict[str, Any]) -> Any:
    raw = event.get("body")
    if not isinstance(raw, str) or not raw:
        raise ApiError(400, "request body is required")
    if len(raw.encode("utf-8")) > MAX_BODY_BYTES:
        raise ApiError(413, f"request body exceeds {MAX_BODY_BYTES} bytes")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(400, "request body is not valid JSON") from exc


def run(action: Callable[[], dict[str, Any]], *, ok_status: int = 200) -> dict[str, Any]:
    try:
        return _response(ok_status, action())
    except ApiError as exc:
        return _response(exc.status, {"error": exc.message})
    except Exception:
        logger.exception("unhandled error")
        return _response(500, {"error": "internal error"})


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def _environ() -> dict[str, str]:
    import os

    return dict(os.environ)
