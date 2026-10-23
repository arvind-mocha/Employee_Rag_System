# Employee RAG System

FastAPI service for employee-document ingestion, retrieval, grounded answers,
and explicitly requested RAG evaluations.

## Start PostgreSQL and Redis

PostgreSQL/PGVector and Redis run as separate services in the same Compose
stack:

```bash
docker compose up -d
docker compose ps
```

No `.env` file is required. PostgreSQL, Redis, PGVector, embeddings, queue
settings, paths, and limits have application defaults in `app/api/config.py`.
The application automatically uses local ports when run on the host and Docker
service names when run in a container.

## Run the API

```bash
uv run uvicorn app.api.main:app --reload
```

The API is available at `http://localhost:8000/api/v1`.

Start the Redis-backed evaluation worker in another terminal:

```bash
uv run employee-rag-worker
```

## Run the frontend

The React/Vite application lives in `frontend/`. Keep the API running on port
`8000`; the Vite development server proxies `/api` requests to it.

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. User mode provides a focused question-and-answer
experience. Developer mode adds model and retrieval controls, retrieved chunks,
request diagnostics, dataset generation, and on-demand evaluation dashboards.

## Configure model providers

Open `http://localhost:5173/settings` and add the API key, model ID, and expiry
period for each provider that this installation should use. The API no longer
loads application configuration, provider keys, or model names from `.env`.

Provider API keys are encrypted with AES-256-GCM before they are stored in the
`provider_credentials` PostgreSQL table. API responses expose only the final
four characters as a key hint. Expired credentials are rejected before use and
deleted automatically; they can also be deleted immediately from the Settings
page.

On first use, local development creates a per-installation encryption key at
`.secrets/credential_encryption_key`. This directory is ignored by Git. Do not
commit, share, or delete that file while encrypted credentials still need to be
read. For a container deployment, mount the same read-only file into the API and
worker containers at the automatically detected path:

```text
/run/secrets/credential_encryption_key
```

Deleting a credential from this application does not revoke it at the model
provider. Revoke the key in the provider dashboard if it may be compromised.

For evaluation jobs, PostgreSQL and Redis must be available and the background
worker shown above must be running.

Ingestion locking and status are stored in Redis, so they survive API restarts
and are shared across API workers. Redis is not used for document or vector
storage; those remain in PostgreSQL/PGVector.

## LLM token usage and approximate cost

Every successful API operation that invokes an LLM returns usage information:

- `POST /answers/generate`: one `usage` object for the answer call
- `POST /evaluations/datasets/generate`: aggregate `usage` for all reference calls
- `POST /evaluations/run`: run-level and provider-level aggregate `usage`, plus
  `answer_usage`, `judge_usage`, and combined `usage` for every result

Example:

```json
{
  "input_tokens": 1000,
  "cached_input_tokens": 0,
  "cache_creation_input_tokens": 0,
  "output_tokens": 120,
  "total_tokens": 1120,
  "token_count_source": "provider",
  "input_cost_per_million_usd": 10.0,
  "cached_input_cost_per_million_usd": 1.0,
  "cache_creation_cost_per_million_usd": 12.5,
  "output_cost_per_million_usd": 50.0,
  "estimated_cost_usd": 0.016,
  "pricing_source": "built_in",
  "currency": "USD"
}
```

Provider token counts are used when available. Otherwise, the API estimates token
counts from text length and labels `token_count_source` as `estimated`. Built-in
standard API rates are included for the configured `gpt-6-astra`,
`claude-opus-5-5`, and `grok-4.7` model IDs.

When pricing is unavailable, token usage is still returned, while
`estimated_cost_usd` is `null` and aggregate results increment `unpriced_calls`.
Costs are estimates and may differ from provider invoices due to pricing tiers,
regional rates, long-context rates, credits, rounding, or provider changes.

## On-demand evaluation

Evaluation is never triggered by ingestion, retrieval, or normal answer requests.
It runs only when one of the evaluation commands or endpoints below is called.

### 1. Create a questions JSONL file

Use one JSON object per line:

```jsonl
{"question":"How many vacation days do employees receive?"}
{"question":"What benefits are available to employees?"}
```

An example is available in `evaluation_questions.example.jsonl`.

### 2. Generate ground truth from the current indexed data

The generator retrieves current evidence and asks the selected reference model to
produce a reference answer, atomic required facts, answerability, and supporting
evidence numbers. It does not use exact string matching.

```bash
uv run employee-rag-generate-eval \
  --dataset employee-policies \
  --questions evaluation_questions.example.jsonl \
  --reference-provider openai \
  --reranking
```

The generated file is stored at
`evaluation_data/employee-policies.jsonl`. Generated evaluation data is ignored by
Git because it can contain document content.

Every record includes:

- A reference answer and atomic required facts
- Retrieved and relevant evidence snapshots
- Chunk, source, evidence, and full-corpus hashes
- Reference provider and model
- Retrieval settings used during generation
- `review_status: "machine_generated"`

Review generated references before treating them as authoritative. After review,
change `review_status` to `"approved"` in the JSONL record.

The same operation is exposed as an API:

```bash
curl -X POST 'http://localhost:8000/api/v1/evaluations/datasets/generate' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_name": "employee-policies",
    "questions": [
      "How many vacation days do employees receive?",
      "What benefits are available to employees?"
    ],
    "reference_provider": "openai",
    "semantic_search": true,
    "keyword_search": true,
    "query_rewriting": false,
    "reranking": true,
    "source_diversity": false,
    "overwrite": false
  }'
```

Set `overwrite` to `true` only when intentionally regenerating a dataset.

You can also upload an existing ground-truth JSONL dataset. The uploaded records
must use the same schema as generated records; the API validates every record,
its evidence hashes, and the `.jsonl` extension before saving it. If
`dataset_name` is omitted, the uploaded filename is used.

```bash
curl -X POST \
  'http://localhost:8000/api/v1/evaluations/datasets/upload' \
  -F 'dataset_file=@employee-policies.jsonl;type=application/x-ndjson' \
  -F 'dataset_name=employee-policies' \
  -F 'overwrite=false'
```

Uploads are limited to 10 MB and 100 records. Structural validation cannot
confirm that reference answers are factually correct or representative. A poor,
incorrect, or unsuitable ground-truth dataset will produce unreliable evaluation
results, so review user-provided and generated datasets before relying on scores.

For background generation, submit the same body to the asynchronous endpoint:

```bash
curl -X POST \
  'http://localhost:8000/api/v1/evaluations/datasets/generate/async' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_name": "employee-policies",
    "questions": ["How many vacation days do employees receive?"],
    "reference_provider": "openai"
  }'
```

It returns HTTP `202` with a `job_id` and `status_url`. Poll that URL:

```bash
curl 'http://localhost:8000/api/v1/evaluations/jobs/JOB_ID'
```

Job states are `queued`, `running`, `completed`, or `failed`. Successful jobs
return the normal dataset-generation response inside `result`.

### 3. Run scores on demand

One request can compare several answer providers against the same ground truth.
The judge provider is selected separately.

```bash
curl -X POST 'http://localhost:8000/api/v1/evaluations/run' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_name": "employee-policies",
    "answer_providers": ["openai", "anthropic", "grok"],
    "judge_provider": "openai",
    "semantic_search": true,
    "keyword_search": true,
    "query_rewriting": false,
    "reranking": true,
    "source_diversity": false
  }'
```

Supported provider names are `huggingface`, `openai`, `anthropic`, and `grok`.
Use `record_ids` in the request to evaluate only selected JSONL records.

The same request can be queued at
`POST /api/v1/evaluations/run/async`. Both asynchronous endpoints require the
`employee-rag-worker` process to be running. The original endpoints remain
synchronous for compatibility.

### Returned scores

Each result includes:

- Retrieval: chunk and source precision@K/recall@K, reciprocal rank,
  average precision@K, and nDCG@K
- Answer: correctness, groundedness, completeness, and relevance
- Citations: judge-scored accuracy and coverage plus deterministic validity
- Behavior: abstention correctness, unsupported claims, and missing facts
- Similarity: lexical and local-embedding semantic similarity
- Operations: retrieval, generation, grading, and total latency
- Cost: per-call token usage and approximate USD cost, with run and provider totals
- Aggregates: averages, pass rate, provider errors, stale references, and
  unreviewed-reference counts

An answer passes when overall, correctness, and groundedness are each at least
`0.80`. A stale record has `passed: null` and is excluded from aggregate scores.
If the indexed corpus changes, regenerate and review the JSONL dataset before using
its scores.
