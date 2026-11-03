# support-docs-qa

A question-answering service over support documentation. Documents are chunked and embedded with Amazon Bedrock, stored in DynamoDB, and retrieved by cosine similarity at query time. Bedrock then answers from the retrieved excerpts and cites them. The whole stack is defined in AWS CDK (Python).

## Architecture

```
POST /documents ──► API Gateway (API key + usage plan)
                         │
                         ▼
                  IngestFunction (Lambda)
                    1. chunk text (800 chars, 100 overlap)
                    2. Titan Text Embeddings v2 ─► 1024-dim vectors   [Bedrock]
                    3. replace chunks for doc_id                       [DynamoDB: Chunks, Documents]

POST /ask ───────► API Gateway (API key + usage plan)
                         │
                         ▼
                  AskFunction (Lambda)
                    1. embed the question                              [Bedrock]
                    2. scan Chunks, rank by cosine, keep top k
                    3. grounded prompt with numbered excerpts
                    4. answer with Claude via Converse                 [Bedrock]
                    5. log query, TTL 30 days                          [DynamoDB: QueryLog]
```

## Where each requirement lives

| Requirement | Location |
|---|---|
| RAG and vector retrieval | `src/rag/chunking.py`, `src/rag/vectors.py`, `src/rag/codec.py`, `src/rag/prompt.py`, `src/rag/service.py` |
| AWS Lambda and API Gateway | `src/handlers/` (handlers), `infra/stack.py` (REST API, usage plan, API key) |
| AWS Bedrock | `src/rag/aws.py` (`BedrockEmbedder`, `BedrockGenerator`), IAM scoping in `infra/stack.py` |
| AWS CDK / infrastructure as code | `app.py`, `infra/stack.py`, `cdk.json`, `tests/test_stack.py` (assertions on the synthesized template) |
| DynamoDB | `infra/stack.py` (three tables, TTL on `QueryLog`), `src/rag/aws.py` (`DynamoRepository`) |

The service logic in `src/rag/service.py` depends only on the interfaces in `src/rag/ports.py`. Tests run it with in-memory fakes, so the core is verified without AWS.

## Local development

Requires Python 3.12 and Node.js 20+ (for the CDK CLI).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

pytest                               # unit tests and CDK template assertions
ruff format . && ruff check .        # format and lint
pyright                              # type check
bandit -r src infra app.py scripts   # security scan
```

## Deploy

1. Pick a Bedrock Claude model that your account can invoke, and set it in `cdk.json` (`generation_model_id`). The repo ships with a placeholder, and `app.py` refuses to synthesize until you replace it. Check what is available with:
   ```bash
   aws bedrock list-foundation-models --by-provider anthropic --region <region>
   ```
   Newer Claude models are invoked through inference profiles (for example `us.anthropic.…`). Set the profile ID here. The stack grants access to both the profile and the underlying foundation model.
2. Enable access to `amazon.titan-embed-text-v2:0` in the Bedrock console for your region if it is not already enabled.
3. Deploy:
   ```bash
   npx aws-cdk bootstrap        # once per account and region
   npx aws-cdk deploy
   ```
4. Fetch the API key and the URL from the stack outputs:
   ```bash
   aws apigateway get-api-key --api-key <ApiKeyId> --include-value --query value --output text
   ```
5. Load the sample corpus:
   ```bash
   API_URL=<ApiUrl> API_KEY=<key> python scripts/seed.py
   ```
6. Ask a question:
   ```bash
   curl -s -X POST "$API_URL/ask" \
     -H "x-api-key: $API_KEY" -H "content-type: application/json" \
     -d '{"question": "How long do card refunds take?"}'
   ```

## API

**POST /documents** `{"doc_id": "returns-policy", "title": "Returns", "text": "..."}`
Chunks, embeds, and stores the document. Re-posting the same `doc_id` replaces it. Returns `{"doc_id", "chunks"}`.

**POST /ask** `{"question": "...", "top_k": 4}`
`top_k` is 1 to 8 and defaults to 4. Returns `{"query_id", "answer", "sources": [{"id", "doc_id", "title", "chunk_index", "score"}]}`. The answer cites sources as `[id]`.

Errors use `{"error": "..."}` with a status code: 400 for invalid input, 413 for oversized input, 503 when no documents are loaded or the corpus is over the scan limit, and 500 for anything else (the response does not include internal detail).

## Design decisions and limits

- **DynamoDB as the vector store.** Each question scans every chunk and ranks them in Lambda. That is simple, has no fixed monthly cost, and needs no extra service. It stops scaling at `MAX_SCAN_CHUNKS` (2000 chunks, about 8 MB of packed vectors read on every question). Beyond that, move retrieval to OpenSearch Serverless or S3 Vectors. The port in `src/rag/ports.py` isolates that change to one adapter.
- **Vectors stored as float32 bytes.** A 1024-dim vector is 4 KB instead of about 20 KB as JSON numbers, which keeps items far under the DynamoDB 400 KB limit.
- **Synchronous ingest with a size cap.** Each chunk is embedded in sequence, so a document is capped at 40 chunks (about 32k characters). That keeps ingest inside the 29-second API Gateway integration limit. Larger documents would need an async pipeline (S3 event, queue, and worker).
- **Re-ingest is idempotent.** A document's old chunks are deleted before the new ones are written. A failure partway through leaves the document partially updated, so re-running the same request repairs it.
- **Prompt injection is reduced, not eliminated.** Excerpts are HTML-escaped inside tags, and the system prompt tells the model to treat them as data. Documents are operator-supplied, so the exposure is limited to content you chose to ingest.
- **Access control is one shared API key with a usage plan** (1000 requests per day, 5 requests per second). That is enough for a demo. Per-user auth would use Cognito or IAM auth on the same API.
- **Least privilege.** The ingest function can invoke only the embedding model. The ask function can invoke the embedding and generation models, and it has read-only access to chunks.
- **Showcase data is disposable.** Tables use `RemovalPolicy.DESTROY`, so `cdk destroy` deletes them. Switch to `RETAIN` before storing real data.

## Status

- Unit tests (77) pass, and the code passes ruff, pyright, and bandit with no findings.
- `cdk synth` succeeds with a test model ID.
- Not yet deployed to a real AWS account from this repo. The first deploy will need a real Claude model ID and Bedrock model access.
