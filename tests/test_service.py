from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from rag.ports import CorpusTooLarge, QueryRecord, StoredChunk
from rag.service import (
    MAX_CHUNKS_PER_DOC,
    MAX_TEXT_CHARS,
    ApiError,
    AskRequest,
    IngestRequest,
    answer_question,
    ingest_document,
    parse_ask,
    parse_ingest,
)

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


class FakeEmbedder:
    def __init__(
        self, table: dict[str, list[float]] | None = None, default: list[float] | None = None
    ):
        self.table = table or {}
        self.default = default or [1.0, 0.0]

    def embed(self, text: str) -> list[float]:
        return self.table.get(text, self.default)


class FailingEmbedder:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("bedrock exploded")


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "Refunds take 5 days [0]."


class FakeRepo:
    def __init__(self, corpus: Sequence[StoredChunk] = (), too_large: bool = False) -> None:
        self.corpus = list(corpus)
        self.too_large = too_large
        self.replaced: list[tuple[str, str, list[StoredChunk]]] = []
        self.queries: list[QueryRecord] = []

    def replace_document(self, doc_id: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        self.replaced.append((doc_id, title, list(chunks)))

    def load_all(self, max_items: int) -> list[StoredChunk]:
        if self.too_large:
            raise CorpusTooLarge("too many")
        return self.corpus

    def record_query(self, record: QueryRecord) -> None:
        self.queries.append(record)


def _chunk(index: int, vector: tuple[float, ...], text: str = "text") -> StoredChunk:
    return StoredChunk(
        doc_id="returns", title="Returns", chunk_index=index, text=text, vector=vector
    )


# --- parsing and validation -------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "not a dict",
        {"title": "t", "text": "x"},
        {"doc_id": "Bad_ID", "title": "t", "text": "x"},
        {"doc_id": "ok", "title": "", "text": "x"},
        {"doc_id": "ok", "title": "t", "text": "   "},
        {"doc_id": "ok", "title": "t", "text": "x" * (MAX_TEXT_CHARS + 1)},
    ],
)
def test_parse_ingest_rejects_bad_input(body: object) -> None:
    with pytest.raises(ApiError) as exc:
        parse_ingest(body)
    assert exc.value.status in (400, 413)


def test_parse_ingest_accepts_valid_input_and_trims_title() -> None:
    req = parse_ingest({"doc_id": "returns-policy", "title": "  Returns ", "text": "Body"})
    assert req == IngestRequest(doc_id="returns-policy", title="Returns", text="Body")


@pytest.mark.parametrize("top_k", [0, 9, True, "3", 2.5])
def test_parse_ask_rejects_bad_top_k(top_k: object) -> None:
    with pytest.raises(ApiError):
        parse_ask({"question": "q", "top_k": top_k})


def test_parse_ask_defaults_top_k() -> None:
    assert parse_ask({"question": " q "}) == AskRequest(question="q", top_k=4)


# --- ingest -----------------------------------------------------------------


def test_ingest_stores_every_chunk_with_its_vector() -> None:
    repo = FakeRepo()
    count = ingest_document(
        IngestRequest(doc_id="returns", title="Returns", text="alpha beta gamma"),
        embedder=FakeEmbedder(default=[0.5, 0.5]),
        repo=repo,
    )
    assert count == 1
    doc_id, title, chunks = repo.replaced[0]
    assert (doc_id, title) == ("returns", "Returns")
    assert chunks[0].vector == (0.5, 0.5)


def test_ingest_rejects_documents_over_chunk_limit() -> None:
    text = " ".join(["word"] * (MAX_CHUNKS_PER_DOC * 200))
    with pytest.raises(ApiError) as exc:
        ingest_document(IngestRequest("big", "Big", text), embedder=FakeEmbedder(), repo=FakeRepo())
    assert exc.value.status == 413


def test_ingest_rejects_text_that_is_empty_after_chunking() -> None:
    with pytest.raises(ApiError) as exc:
        ingest_document(IngestRequest("x", "X", "  \n "), embedder=FakeEmbedder(), repo=FakeRepo())
    assert exc.value.status == 400


# --- ask --------------------------------------------------------------------


def test_ask_ranks_the_most_similar_chunk_first_and_cites_it() -> None:
    corpus = [_chunk(0, (0.0, 1.0), "shipping info"), _chunk(1, (1.0, 0.0), "refund info")]
    gen, repo = FakeGenerator(), FakeRepo(corpus)
    result = answer_question(
        AskRequest(question="refund?", top_k=1),
        embedder=FakeEmbedder(default=[1.0, 0.0]),
        generator=gen,
        repo=repo,
        now=NOW,
        query_id="q-1",
    )
    assert [h.chunk.chunk_index for h in result.hits] == [1]
    assert "refund info" in gen.calls[0][1]
    assert "shipping info" not in gen.calls[0][1]
    assert result.answer == "Refunds take 5 days [0]."


def test_ask_logs_query_with_ttl_in_the_future() -> None:
    repo = FakeRepo([_chunk(0, (1.0, 0.0))])
    answer_question(
        AskRequest(question="q", top_k=1),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
        repo=repo,
        now=NOW,
        query_id="q-2",
    )
    record = repo.queries[0]
    assert record.source_keys == ["returns#0"]
    assert record.expires_at > int(NOW.timestamp())


def test_ask_with_empty_corpus_is_503() -> None:
    with pytest.raises(ApiError) as exc:
        answer_question(
            AskRequest("q", 1),
            embedder=FakeEmbedder(),
            generator=FakeGenerator(),
            repo=FakeRepo([]),
            now=NOW,
            query_id="q-3",
        )
    assert exc.value.status == 503


def test_ask_over_scan_limit_is_503() -> None:
    with pytest.raises(ApiError) as exc:
        answer_question(
            AskRequest("q", 1),
            embedder=FakeEmbedder(),
            generator=FakeGenerator(),
            repo=FakeRepo(too_large=True),
            now=NOW,
            query_id="q-4",
        )
    assert exc.value.status == 503


def test_embedding_failure_propagates_so_the_handler_can_return_500() -> None:
    with pytest.raises(RuntimeError):
        answer_question(
            AskRequest("q", 1),
            embedder=FailingEmbedder(),
            generator=FakeGenerator(),
            repo=FakeRepo([_chunk(0, (1.0, 0.0))]),
            now=NOW,
            query_id="q-5",
        )
