"""Pack embedding vectors as little-endian float32 bytes for DynamoDB Binary attributes.

Why: a 1024-dim vector is 4 KB packed versus roughly 20 KB as JSON numbers, and
DynamoDB items are capped at 400 KB. Float32 precision is ample for cosine ranking.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

_FLOAT32_BYTES = 4


def encode_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(data: bytes) -> tuple[float, ...]:
    if len(data) % _FLOAT32_BYTES != 0:
        raise ValueError(f"vector payload length {len(data)} is not a multiple of 4")
    count = len(data) // _FLOAT32_BYTES
    return struct.unpack(f"<{count}f", data)
