# API reference

The FastAPI base path is `/api/v1`. During local development the API is
available at `http://localhost:8000`, with interactive OpenAPI documentation at
`http://localhost:8000/docs`.

## Endpoint catalog

### Data and ingestion

| Method | Endpoint | Success | Purpose |
| --- | --- | --- | --- |
| `POST` | `/data/upload` | `201` | Validate, extract, classify, and store an uploaded ZIP |
| `POST` | `/ingestion/start` | `202` | Reserve and start one ingestion run |
| `GET` | `/ingestion/status` | `200` | Return the latest ingestion state and timestamps |
| `DELETE` | `/data/vector-database` | `200` | Erase indexed vectors while preserving source files and credentials |

Ingestion states are `idle`, `pending`, `running`, `completed`, and `failed`.

### Retrieval and answers

| Method | Endpoint | Success | Purpose |
| --- | --- | --- | --- |
| `GET` | `/retrieval/search` | `200` | Retrieve relevant chunks without calling an LLM |
| `POST` | `/answers/generate` | `200` | Let the LLM call employee-knowledge search when needed, then answer |

Retrieval options are:

- `semantic_search`
- `keyword_search`
- `query_rewriting`
- `reranking`
- `source_diversity`

At least one of semantic or keyword search must be enabled.

### Provider settings

| Method | Endpoint | Success | Purpose |
| --- | --- | --- | --- |
| `GET` | `/settings/providers` | `200` | List masked active credentials and configuration status for all providers |
| `POST` | `/settings/providers` | `201` | Encrypt and save or replace a credential |
| `DELETE` | `/settings/providers/{provider}` | `204` | Delete a saved provider credential |

Provider names are `gemini`, `openai`, `anthropic`, and `grok`.

The existing GET response retains `credentials` and adds `providers`, with one
entry per supported provider: `provider`, `model_name`, `expires_at`, `status`,
and `message`. Status is `configured`, `not_configured`, `expired`, or
`unavailable`. `configured` indicates saved, non-expired configuration, not
provider-side verification. No key, ciphertext, nonce, or key hint is included
in the status entries. The existing `credentials` metadata still includes masked
key hints. An unavailable credential store returns `503`; clients must display
Configuration Unavailable rather than trusting cached configuration.


### Evaluation

| Method | Endpoint | Success | Purpose |
| --- | --- | --- | --- |
| `POST` | `/evaluations/datasets/upload` | `201` | Validate and persist one or multiple JSONL files |
| `GET` | `/evaluations/datasets` | `200` | List metadata for all datasets |
| `GET` | `/evaluations/reference/example` | `200` | Download the tracked five-record JSONL example |
| `GET` | `/evaluations/reference/schema` | `200` | Record JSON Schema, categories, and configured limits |
| `GET` | `/evaluations/datasets/{dataset_id}` | `200` | Metadata and paginated record preview (`offset`, `limit`, max 100) |
| `DELETE` | `/evaluations/datasets/{dataset_id}` | `200` | Delete one specific dataset |
| `POST` | `/evaluations/run` | `200` | Evaluate explicitly selected dataset IDs synchronously |
| `POST` | `/evaluations/run/async` | `202` | Queue selected datasets for evaluation |
| `GET` | `/evaluations/jobs/{job_id}` | `200` | Read job status, progress, result, or error |

Asynchronous job states include `queued`, `running`, `completed`, `failed`,
`deferred`, `scheduled`, `stopped`, and `canceled`.

## Examples

### Upload source data

```bash
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F 'zip_file=@company-data.zip;type=application/zip'
```

The response lists the structured and unstructured filenames that were stored,
plus `ignored_files` and `renamed_files`. Unsupported members are ignored when
at least one supported file remains. Additional ZIPs are additive; a filename
already present in either source-data directory is preserved and the incoming
file receives a numeric suffix such as `_1`. File-level replacement is not
supported.

### Start and poll ingestion

```bash
curl -X POST http://localhost:8000/api/v1/ingestion/start
curl http://localhost:8000/api/v1/ingestion/status
```

Starting a second concurrent run returns `409 Conflict`.

### Erase indexed vectors

```bash
curl -X DELETE http://localhost:8000/api/v1/data/vector-database
```

This operation shares the ingestion lock, drops only the configured vector
table, and resets readiness to `idle`. Uploaded source files, evaluation files,
and encrypted provider credentials are preserved. Run ingestion again to
rebuild the index. A concurrent ingestion returns `409 Conflict`.

### Retrieve chunks

```bash
curl --get http://localhost:8000/api/v1/retrieval/search \
  --data-urlencode 'question=What is the remote work policy?' \
  --data 'semantic_search=true' \
  --data 'keyword_search=true' \
  --data 'reranking=true'
```

### Generate a tool-directed answer

```bash
curl -X POST http://localhost:8000/api/v1/answers/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is the remote work policy?",
    "semantic_search": true,
    "keyword_search": true,
    "query_rewriting": false,
    "reranking": true,
    "source_diversity": true,
    "gemini": false,
    "openai": true,
    "anthropic": false,
    "grok": false
  }'
```

At most one provider switch may be true. If none is selected, Google Gemini is
the backend default. The configured model must support tool calling. The
response field `retrieval_used` reports whether the model called
`search_employee_knowledge`; `sources` and `applied_strategies` are empty when
the model answers directly. Retrieval settings apply only when the tool is used.

### Save a provider credential

Use the Settings page when possible so the key is not left in shell history.
The underlying request format is:

```json
{
  "provider": "openai",
  "model_name": "provider-model-id",
  "api_key": "REPLACE_WITH_KEY",
  "expires_in_hours": 24
}
```

`expires_in_hours` must be between `0.25` and `8760`. The response never returns
the plaintext key.

### Download the evaluation template

```bash
curl -o evaluation_dataset.example.jsonl \
  http://localhost:8000/api/v1/evaluations/reference/example
```

The response uses `application/x-ndjson` and an attachment filename. It is the
same file used by the frontend preview and copy action. The schema endpoint
returns `record_schema`, `categories`, `maximum_file_size_bytes`,
`maximum_records`, and `maximum_upload_files` from runtime configuration.

### Upload evaluation files

```bash
curl -X POST http://localhost:8000/api/v1/evaluations/datasets/upload \
  -F 'files=@examples/evaluation_dataset.example.jsonl' \
  -F 'files=@another-dataset.jsonl'
```

Multipart `files` accepts one or more files. The earlier `dataset_file` field is
also accepted for a single file. Dataset naming/overwrite form fields are no
longer used: every accepted file gets a unique `dataset_id`. The response contains
`datasets` (metadata for accepted files) and `errors` (filename and validation
message for rejected files). A mixed batch returns `201` with both lists; a wholly
invalid batch returns `422` with filenames and errors in `detail`. Invalid record
messages include physical line numbers. No invalid file is partially saved.

Each metadata object includes `dataset_id`, `filename`, `record_count`,
`size_bytes`, `uploaded_at`, `validation_status`, `validation_error`, and
`dataset_sha256`. Use the stable ID for preview, deletion, and evaluation.
See [evaluation schema and examples](evaluation.md#jsonl-schema).

### Queue an evaluation run

```bash
curl -X POST http://localhost:8000/api/v1/evaluations/run/async \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_ids": ["<first-dataset-id>", "<second-dataset-id>"],
    "answer_providers": ["openai", "anthropic"],
    "judge_provider": "openai",
    "semantic_search": true,
    "keyword_search": true,
    "query_rewriting": false,
    "reranking": true,
    "source_diversity": false
  }'
```

The selected IDs must be nonempty, unique, and available. Optional `record_ids`
restricts cases within each selected file and must exist in every selected file.
Provider and retrieval settings apply to all selected datasets. Both run endpoints check
that each selected answer/judge provider has a saved non-expired key and model;
the worker repeats this check at execution. Missing/expired credentials or model
configuration produce `422` with the affected provider and a Settings action;
credential-store failures produce `503`. A failed preflight queues no work.

Unselected files are never evaluated. The async response contains `job_id` and `status_url`;
poll that URL to retrieve `completed_cases`, `total_cases`, `current_dataset_id`,
status, and result. Both synchronous responses and completed async `result`
contain `run_id`, `datasets` (independent per-file runs), and `aggregate` (combined
case-weighted provider summaries and results). Cases include `dataset_id` and
`dataset_name` as well as `record_id`. Missing retrieval ground truth yields null
retrieval metrics. Queued jobs reject files changed or deleted before processing.

Dataset generation endpoints and the generation CLI have been removed.

## Limits and validation

| Input | Default limit |
| --- | --- |
| Source ZIP | 50 MB compressed |
| Extracted source data | 200 MB total |
| Files in a ZIP | 1,000 |
| Evaluation upload | 10 MiB per file, 5 files per upload |
| Evaluation records | 100 per file |
| Datasets selected per run | 100 |
| Question length | 2,000 characters |
| Answer providers in one evaluation | 4 |

FastAPI returns validation details for malformed requests. Service failures use
safe messages without returning database URLs, decrypted keys, or provider
secrets.

Evaluation limits are configurable with `MAX_EVALUATION_DATASET_SIZE` (bytes),
`MAX_EVALUATION_RECORDS`, and `MAX_EVALUATION_UPLOAD_FILES` for API and worker.
