import itertools
import random

import pytest

from rag.chunking import chunk_text


def test_empty_and_whitespace_return_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("hello world", max_chars=800, overlap=100)
    assert [c.text for c in chunks] == ["hello world"]
    assert chunks[0].index == 0


def test_chunks_respect_max_chars_and_are_indexed_in_order() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, max_chars=200, overlap=20)
    assert len(chunks) > 1
    assert all(0 < len(c.text) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_every_word_survives_chunking() -> None:
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, max_chars=120, overlap=15)
    joined = " ".join(c.text for c in chunks)
    for i in range(500):
        assert f"w{i}" in joined.split()


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"tok{i}" for i in range(300))
    chunks = chunk_text(text, max_chars=100, overlap=30)
    for a, b in itertools.pairwise(chunks):
        assert set(a.text.split()) & set(b.text.split())


def test_no_spaces_still_terminates_with_bounded_chunks() -> None:
    chunks = chunk_text("x" * 1000, max_chars=100, overlap=99)
    assert chunks
    assert all(len(c.text) <= 100 for c in chunks)


@pytest.mark.parametrize("seed", range(20))
def test_random_text_terminates_and_stays_bounded(seed: int) -> None:
    rng = random.Random(seed)
    alphabet = "ab  \n.x"
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 3000)))
    max_chars = rng.randint(1, 50)
    overlap = rng.randint(0, max_chars - 1)
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    assert all(0 < len(c.text) <= max_chars for c in chunks)


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_raise(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap=overlap)
