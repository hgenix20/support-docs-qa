import json
from typing import Any

import pytest

from handlers import ask as ask_module
from handlers import ingest as ingest_module
from handlers.common import MAX_BODY_BYTES, Deps, parse_json_body, run
from rag.ports import StoredChunk
from rag.service import ApiError
from tests.test_service import FailingEmbedder, FakeEmbedder, FakeGenerator, FakeRepo


def _event(body: Any) -> dict[str, Any]:
    return {"body": body if isinstance(body, str) else json.dumps(body)}


def _deps(embedder: Any, repo: FakeRepo | None = None) -> Deps:
    return Deps(embedder=embedder, generator=FakeGenerator(), repo=repo or FakeRepo())


def test_parse_json_body_rejects_missing_invalid_and_oversized_bodies() -> None:
    with pytest.raises(ApiError) as missing:
        parse_json_body({})
    assert missing.value.status == 400
    with pytest.raises(ApiError) as invalid:
        parse_json_body(_event("{not json"))
    assert invalid.value.status == 400
    with pytest.raises(ApiError) as big:
        parse_json_body(_event("x" * (MAX_BODY_BYTES + 1)))
    assert big.value.status == 413


def test_run_maps_api_errors_to_their_status() -> None:
    def boom() -> dict[str, Any]:
        raise ApiError(418, "teapot")

    response = run(boom)
    assert response["statusCode"] == 418
    assert json.loads(response["body"]) == {"error": "teapot"}


def test_run_hides_internal_error_details() -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("secret account 123456789012 detail")

    response = run(boom)
    assert response["statusCode"] == 500
    assert "123456789012" not in response["body"]
    assert json.loads(response["body"]) == {"error": "internal error"}


def test_ingest_handler_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo()
    monkeypatch.setattr(ingest_module, "get_deps", lambda: _deps(FakeEmbedder(), repo))
    response = ingest_module.handler(
        _event({"doc_id": "returns", "title": "Returns", "text": "Refunds take 5 days."}), None
    )
    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"doc_id": "returns", "chunks": 1}
    assert repo.replaced[0][0] == "returns"


def test_ask_handler_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = [
        StoredChunk("returns", "Returns", 0, "Refunds take 5 days.", (1.0, 0.0)),
        StoredChunk("shipping", "Shipping", 0, "Ships in 2 days.", (0.0, 1.0)),
    ]
    repo = FakeRepo(corpus)
    monkeypatch.setattr(
        ask_module, "get_deps", lambda: _deps(FakeEmbedder(default=[1.0, 0.0]), repo)
    )
    response = ask_module.handler(_event({"question": "refund?", "top_k": 1}), None)
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["answer"] == "Refunds take 5 days [0]."
    assert body["sources"][0]["doc_id"] == "returns"
    assert body["sources"][0]["id"] == 0
    assert repo.queries[0].query_id == body["query_id"]


def test_ask_handler_returns_500_without_leaking_when_embedding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeRepo([StoredChunk("returns", "Returns", 0, "t", (1.0, 0.0))])
    monkeypatch.setattr(ask_module, "get_deps", lambda: _deps(FailingEmbedder(), repo))
    response = ask_module.handler(_event({"question": "q"}), None)
    assert response["statusCode"] == 500
    assert "bedrock exploded" not in response["body"]
