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

Ingestion states are `idle`, `pending`, `running`, `completed`, and `failed`.

### Retrieval and answers

| Method | Endpoint | Success | Purpose |
| --- | --- | --- | --- |
| `GET` | `/retrieval/search` | `200` | Retrieve relevant chunks without calling an LLM |
| `POST` | `/answers/generate` | `200` | Retrieve context and generate one grounded answer |

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
| `GET` | `/settings/providers` | `200` | List active credentials as masked metadata |
| `POST` | `/settings/providers` | `201` | Encrypt and save or replace a credential |
| `DELETE` | `/settings/providers/{provider}` | `204` | Delete a saved provider credential |

Provider names are `huggingface`, `openai`, `anthropic`, and `grok`.

### Evaluation

| Method | Endpoint | Success | Purpose |
| --- | --- | --- | --- |
| `POST` | `/evaluations/datasets/generate` | `201` | Generate ground truth synchronously |
| `POST` | `/evaluations/datasets/upload` | `201` | Validate and save an evaluation JSONL file |
| `POST` | `/evaluations/run` | `200` | Run an evaluation synchronously |
| `POST` | `/evaluations/datasets/generate/async` | `202` | Queue ground-truth generation |
| `POST` | `/evaluations/run/async` | `202` | Queue an evaluation run |
| `GET` | `/evaluations/jobs/{job_id}` | `200` | Read job status, result, or error |

Asynchronous job states include `queued`, `running`, `completed`, `failed`,
`deferred`, `scheduled`, `stopped`, and `canceled`.

## Examples

### Upload source data

```bash
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F 'zip_file=@company-data.zip;type=application/zip'
```

The response lists the structured and unstructured filenames that were stored.

### Start and poll ingestion

```bash
curl -X POST http://localhost:8000/api/v1/ingestion/start
curl http://localhost:8000/api/v1/ingestion/status
```

Starting a second concurrent run returns `409 Conflict`.

### Retrieve chunks

```bash
curl --get http://localhost:8000/api/v1/retrieval/search \
  --data-urlencode 'question=What is the remote work policy?' \
  --data 'semantic_search=true' \
  --data 'keyword_search=true' \
  --data 'reranking=true'
```

### Generate a grounded answer

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
    "hf": false,
    "openai": true,
    "anthropic": false,
    "grok": false
  }'
```

At most one provider switch may be true. If none is selected, Hugging Face is
the backend default.

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

### Queue ground-truth generation

```bash
curl -X POST \
  http://localhost:8000/api/v1/evaluations/datasets/generate/async \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_name": "employee-policies",
    "questions": ["How many vacation days do employees receive?"],
    "reference_provider": "openai",
    "semantic_search": true,
    "keyword_search": true,
    "query_rewriting": false,
    "reranking": true,
    "source_diversity": false,
    "overwrite": false
  }'
```

The response contains a `job_id` and `status_url`. Poll the returned URL until
the job completes or fails.

### Queue an evaluation run

```bash
curl -X POST http://localhost:8000/api/v1/evaluations/run/async \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_name": "employee-policies",
    "answer_providers": ["openai", "anthropic"],
    "judge_provider": "openai",
    "semantic_search": true,
    "keyword_search": true,
    "query_rewriting": false,
    "reranking": true,
    "source_diversity": false
  }'
```

## Limits and validation

| Input | Default limit |
| --- | --- |
| Source ZIP | 50 MB compressed |
| Extracted source data | 200 MB total |
| Files in a ZIP | 1,000 |
| Evaluation upload | 10 MB |
| Evaluation records or generated questions | 100 |
| Question length | 2,000 characters |
| Answer providers in one evaluation | 4 |

FastAPI returns validation details for malformed requests. Service failures use
safe messages without returning database URLs, decrypted keys, or provider
secrets.
