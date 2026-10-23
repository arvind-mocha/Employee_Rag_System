# Employee RAG System

FastAPI service for employee-document ingestion, retrieval, grounded answers,
and explicitly requested RAG evaluations.

## Run the API

```bash
uv run uvicorn app.api.main:app --reload
```

The API is available at `http://localhost:8000/api/v1`.

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

### Returned scores

Each result includes:

- Retrieval: chunk and source precision@K/recall@K, reciprocal rank,
  average precision@K, and nDCG@K
- Answer: correctness, groundedness, completeness, and relevance
- Citations: judge-scored accuracy and coverage plus deterministic validity
- Behavior: abstention correctness, unsupported claims, and missing facts
- Similarity: lexical and local-embedding semantic similarity
- Operations: retrieval, generation, grading, and total latency
- Aggregates: averages, pass rate, provider errors, stale references, and
  unreviewed-reference counts

An answer passes when overall, correctness, and groundedness are each at least
`0.80`. A stale record has `passed: null` and is excluded from aggregate scores.
If the indexed corpus changes, regenerate and review the JSONL dataset before using
its scores.
