import pytest

from rag.codec import decode_vector, encode_vector
from rag.vectors import cosine, rank_top_k


def test_cosine_identical_is_one() -> None:
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one() -> None:
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_zero_not_nan() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine([1.0], [1.0, 2.0])


def test_rank_top_k_orders_by_score_and_bounds_k() -> None:
    query = [1.0, 0.0]
    candidates = [("far", [0.0, 1.0]), ("best", [1.0, 0.0]), ("mid", [1.0, 1.0])]
    ranked = rank_top_k(query, candidates, k=2)
    assert [item for item, _ in ranked] == ["best", "mid"]


def test_rank_top_k_with_fewer_candidates_than_k() -> None:
    assert len(rank_top_k([1.0], [("a", [1.0])], k=5)) == 1


def test_rank_top_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        rank_top_k([1.0], [("a", [1.0])], k=0)


def test_codec_round_trip_within_float32_precision() -> None:
    vector = [0.1, -0.25, 3.5, 1e-3]
    decoded = decode_vector(encode_vector(vector))
    assert decoded == pytest.approx(vector, rel=1e-6)


def test_codec_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError):
        decode_vector(b"\x00\x00\x00")
