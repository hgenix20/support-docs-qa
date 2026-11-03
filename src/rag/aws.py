"""AWS adapters for the ports: Bedrock (embed, generate) and DynamoDB (storage).

This is the only module that talks to AWS. Settings come from environment
variables that the CDK stack sets; a missing one fails at cold start, not later.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from boto3.dynamodb.types import Binary

from rag.codec import decode_vector, encode_vector
from rag.ports import CorpusTooLarge, QueryRecord, StoredChunk

EMBED_DIMENSIONS = 1024
GENERATION_MAX_TOKENS = 600


@dataclass(frozen=True)
class Settings:
    chunks_table: str
    documents_table: str
    query_log_table: str
    embed_model_id: str
    generation_model_id: str
    max_scan_chunks: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        def required(name: str) -> str:
            value = env.get(name)
            if not value:
                raise RuntimeError(f"missing required environment variable {name}")
            return value

        max_scan = int(required("MAX_SCAN_CHUNKS"))
        if max_scan <= 0:
            raise RuntimeError("MAX_SCAN_CHUNKS must be positive")
        return cls(
            chunks_table=required("CHUNKS_TABLE"),
            documents_table=required("DOCUMENTS_TABLE"),
            query_log_table=required("QUERY_LOG_TABLE"),
            embed_model_id=required("EMBED_MODEL_ID"),
            generation_model_id=required("GENERATION_MODEL_ID"),
            max_scan_chunks=max_scan,
        )


class BedrockEmbedder:
    def __init__(self, client: Any, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    def embed(self, text: str) -> list[float]:
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps({"inputText": text, "dimensions": EMBED_DIMENSIONS, "normalize": True}),
            contentType="application/json",
            accept="application/json",
        )
        vector = json.loads(response["body"].read())["embedding"]
        if len(vector) != EMBED_DIMENSIONS:
            raise RuntimeError(
                f"embedding model returned {len(vector)} dims, expected {EMBED_DIMENSIONS}"
            )
        return [float(x) for x in vector]


class BedrockGenerator:
    def __init__(self, client: Any, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    def generate(self, system: str, user: str) -> str:
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": GENERATION_MAX_TOKENS, "temperature": 0.0},
        )
        blocks = response["output"]["message"]["content"]
        text = "".join(block.get("text", "") for block in blocks)
        if not text:
            raise RuntimeError("generation model returned no text")
        return text


class DynamoRepository:
    """Chunks keyed by (doc_id, chunk_index). Documents keyed by doc_id. QueryLog keyed by query_id.

    QueryLog rows carry expires_at, which DynamoDB TTL uses to delete them.
    """

    def __init__(self, resource: Any, settings: Settings) -> None:
        self._chunks = resource.Table(settings.chunks_table)
        self._documents = resource.Table(settings.documents_table)
        self._query_log = resource.Table(settings.query_log_table)
        self._max_scan = settings.max_scan_chunks

    def replace_document(self, doc_id: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        # Delete and put in separate batches: BatchWriteItem rejects a request that
        # touches the same key twice, and an old chunk can share a key with a new one.
        existing_keys = self._existing_keys(doc_id)
        if existing_keys:
            with self._chunks.batch_writer() as batch:
                for key in existing_keys:
                    batch.delete_item(Key=key)
        now = datetime.now(UTC).isoformat()
        with self._chunks.batch_writer() as batch:
            for chunk in chunks:
                batch.put_item(
                    Item={
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "title": chunk.title,
                        "text": chunk.text,
                        "embedding": Binary(encode_vector(chunk.vector)),
                        "created_at": now,
                    }
                )
        self._documents.put_item(
            Item={"doc_id": doc_id, "title": title, "chunk_count": len(chunks), "ingested_at": now}
        )

    def load_all(self, max_items: int) -> list[StoredChunk]:
        rows: list[StoredChunk] = []
        kwargs: dict[str, Any] = {}
        while True:
            page = self._chunks.scan(**kwargs)
            for item in page["Items"]:
                rows.append(
                    StoredChunk(
                        doc_id=item["doc_id"],
                        title=item["title"],
                        chunk_index=int(item["chunk_index"]),
                        text=item["text"],
                        vector=decode_vector(bytes(item["embedding"].value)),
                    )
                )
            if len(rows) > max_items:
                raise CorpusTooLarge(f"more than {max_items} chunks")
            if "LastEvaluatedKey" not in page:
                return rows
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def record_query(self, record: QueryRecord) -> None:
        self._query_log.put_item(
            Item={
                "query_id": record.query_id,
                "question": record.question,
                "answer": record.answer,
                "source_keys": record.source_keys,
                "created_at": record.created_at.isoformat(),
                "expires_at": record.expires_at,
            }
        )

    def _existing_keys(self, doc_id: str) -> list[dict[str, Any]]:
        keys: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "doc_id = :d",
            "ExpressionAttributeValues": {":d": doc_id},
            "ProjectionExpression": "doc_id, chunk_index",
        }
        while True:
            page = self._chunks.query(**kwargs)
            keys.extend(
                {"doc_id": i["doc_id"], "chunk_index": i["chunk_index"]} for i in page["Items"]
            )
            if "LastEvaluatedKey" not in page:
                return keys
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
