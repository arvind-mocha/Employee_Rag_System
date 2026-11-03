# RAG quality evaluation

Evaluation uses JSONL files supplied by users. The application does not create
questions, reference answers, or ground-truth datasets. Evaluation runs only when
explicitly requested and is separate from document ingestion and employee chat.

## Workflow

1. Complete document ingestion and configure answer and judge provider credentials.
2. Start Redis and the worker (`uv run employee-rag-worker`), or use Docker Compose.
3. Open **Developer Mode → Evaluation**.
4. Drop one or more `.jsonl` files into the dataset library or select **Upload datasets**.
5. Inspect validation feedback and preview the stored records.
6. Check the files you want to evaluate; **Select All** includes valid files only.
   **Clear Selection** deselects every file. Uploading more files preserves the library.
7. Choose answer providers, a judge provider, and retrieval strategies. At least
   one of semantic or keyword search must be enabled.
8. Click **Run Evaluation**. Only the selected dataset IDs are sent to the worker.
9. Use **Results for** to inspect the aggregate or an individual dataset.

Files with the same filename receive different stable dataset IDs and never
replace one another. Deleting one library entry removes only that dataset.
Selection and the latest accepted job are retained in browser local storage.
Jobs continue during navigation and page reloads; the browser resumes polling.

## Provider configuration and preflight validation

The **LLM Provider Configuration** section displays Google Gemini, OpenAI,
Anthropic, and Grok (xAI), including their saved model and configuration status.
It uses the existing `GET /api/v1/settings/providers` endpoint; no API keys are
sent to the browser. Statuses are:

| Label | Meaning |
| --- | --- |
| Configured (green) | A saved, non-expired key and nonblank model are present |
| API Key Not Configured (red) | No usable saved key or model; the message identifies what to configure |
| API Key Expired (amber) | The saved credential has expired |
| Configuration Unavailable | The backend could not report status; evaluations remain blocked |

Configured describes local saved state, not external provider verification.
Use **Configure in Settings** to navigate explicitly. The page shows separate
loading and retrieval-error states and offers retry on failure. It shares the
Settings credential query cache, refreshes on returning to the page/window and
every 15 seconds, and refreshes at the next known expiry. Background status
failures override cached configured status. Dataset and provider selections are
preserved when credentials become unavailable, including navigation to Settings.
An unavailable answer provider already selected may be deselected, but cannot
be newly selected. Unavailable judges cannot be newly selected.

Immediately before submission, the frontend fetches credential status again.
The synchronous and asynchronous endpoints validate all selected answer and judge
providers, including saved model, expiry, and decryptability. The worker repeats
the validation when execution starts. Invalid selections return an actionable,
provider-specific error before queuing a job. After local validation, preflight
requires completed ingestion, valid retrieval limits, and a registered worker for
asynchronous runs. It makes a small answer request for every selected answer
provider and a separate structured-output request for the judge, even when both
roles use the same provider. These probes consume provider tokens and can incur
charges. Each probe uses a 20-second timeout without retries; the worker repeats
preflight when execution starts to catch configuration changes after submission.
Authentication, quota, model, or judge-format failures return an actionable `422`;
transient provider and readiness failures return `503`. Keys never appear in job
payloads or results. Model access, operation support, permissions, and available
quota are verified only to the extent of these real probe requests.

Expired secrets are cleared when read and by the periodic cleanup task. Only
provider/model/expiry metadata remains, enabling the amber expired status.
Replacement updates the same saved provider; explicit deletion removes its
metadata as well. Credentials deleted by older application versions cannot be
retroactively distinguished from credentials that were never saved.

## Example JSONL Template

The **JSONL Developer Reference** section provides **View Example**,
**Download Example JSONL**, **Copy JSONL**, and **View schema documentation**.
The formatted preview and raw copyable source use the same backend download
endpoint and tracked file:
`examples/evaluation_dataset.example.jsonl`.
It contains five fictional Cedar Lane Labs cases: direct fact, procedure,
multiple facts (`spanning`), numerical reasoning, and an unanswerable question.
It demonstrates expected keywords, facts, evidence content, and optional metadata
without document IDs, chunk IDs, fingerprints, or model provenance. The numerical
case omits evidence deliberately to demonstrate unavailable retrieval metrics.
Replace fictional policies with your own reviewed references for meaningful scores.

`GET /api/v1/evaluations/reference/example` returns that file verbatim as a UTF-8
attachment. `GET /api/v1/evaluations/reference/schema` returns the actual record
JSON Schema, supported categories, and configured file/record/upload limits.
The existing upload schema and evaluation scoring rules are unchanged.

## JSONL schema

Use UTF-8 (an optional UTF-8 BOM is accepted), with one JSON object per nonempty
line. Blank lines are ignored. Every record requires two nonblank strings:

```jsonl
{"question":"How many vacation days do employees receive?","reference_answer":"Employees receive 15 vacation days annually."}
{"id":"unknown-office-code","question":"What is the office access code?","reference_answer":"The documents do not provide this information.","answerable":false}
```

These sample facts are illustrative; supply references that match your corpus.
A fuller example is available at [examples/evaluation_dataset.example.jsonl](../examples/evaluation_dataset.example.jsonl).

| Field | Requirement / default |
| --- | --- |
| `question` | Required string, at most 2,000 characters |
| `reference_answer` | Required nonblank string, including an expected abstention for unanswerable cases |
| `id` | Optional nonblank string, unique within the file; defaults to `line-N` using the physical line number |
| `answerable` | Optional boolean, defaults to `true`; use `false` for unanswerable questions |
| `category` | Optional; defaults to `direct_fact`. Other values: `temporal`, `spanning`, `relationship`, `numerical`, `comparative`, `holistic`, `procedural`, `unanswerable` |
| `keywords` / `expected_keywords` | Optional list of strings |
| `required_facts` | Optional list of atomic reference facts |
| `relevant_evidence` / `reference_evidence` | Optional list of evidence objects, described below |
| `metadata` | Optional object for user-defined metadata |
| `review_status` | Defaults to `user_provided`; legacy `approved` and `machine_generated` values remain accepted |

An `answerable:false` record is categorized as `unanswerable`; it must have empty
`required_facts` and relevant evidence. Answerable records cannot use the
`unanswerable` category. Keywords are cleared for unanswerable records.

Each reference evidence object requires a nonempty `content` string. Optional
fields include `position` (positive integer, default 1), `chunk_id`, `source_file`,
`source_type`, `section`, `chunk_index`, `content_sha256`, and `source_sha256`.
The parser calculates a missing content hash from supplied evidence; this hashes
user-provided content and does not invent ground truth. Supplied hashes and
fingerprints are verified. Chunk metrics compare exact evidence content hashes;
source precision/recall additionally need `source_file` values. No reference
evidence means retrieval metrics are `null` and the result explains why.

Existing schema versions 1 and 2 remain accepted, including `retrieved_evidence`,
`evidence_fingerprint`, `corpus_fingerprint`, `generated_at`, `reference_provider`,
`reference_model`, `reference_usage`, and `retrieval`. These provenance fields
are optional; no reference provider is invoked to fill them. When retrieved and
relevant evidence are both supplied, relevant evidence must belong to the
retrieved snapshots, preserving validation of legacy records.

## Validation and persistence

Malformed JSON, non-object records, missing required fields, duplicate record
IDs, contradictory answerability, and evidence hash mismatches are rejected
with the physical line number. Invalid UTF-8, empty files, unsupported extensions,
and unsafe filenames are also rejected. Invalid files are never partially saved.
A mixed upload batch retains valid files and reports each invalid file separately.

| Environment variable | Default |
| --- | --- |
| `MAX_EVALUATION_DATASET_SIZE` | 10,485,760 bytes (10 MiB) per file |
| `MAX_EVALUATION_RECORDS` | 100 per file |
| `MAX_EVALUATION_UPLOAD_FILES` | 5 files per upload |

Set the same limits for API and worker. Docker Compose forwards these variables
from the shell or `.env`. If raising total upload sizes, also adjust nginx's
`client_max_body_size` (55 MB by default).

Each new upload is stored as `evaluation_data/<dataset_id>/dataset.jsonl` with
`metadata.json`. The original bytes, original filename, record count, byte size,
upload time, and validation state remain available after application restarts.
The API and worker share the persistent `evaluation_data/` Docker bind mount.
Existing flat `evaluation_data/<name>.jsonl` files remain discoverable, with their
original stem as their stable ID and filesystem modification time as upload time.

## Evaluation and reporting

All selected files are validated before evaluation starts. Queued jobs retain
the selected IDs and content hashes. If a selected file disappears or changes
before processing, the job fails explicitly; it never substitutes another file.
Each selected dataset runs independently. Every case carries its dataset ID and
filename, so repeated filenames or record IDs across files remain distinct.

The response includes `datasets` (individual runs) and `aggregate` (one combined
run). Aggregate provider averages, pass rates, latency, token counts, and cost
are calculated from all individual cases, rather than averaging dataset averages.
Unavailable metrics are excluded from averages. Errored cases are reported
separately from passed and failed cases.

| Category | Metrics |
| --- | --- |
| Retrieval | Precision@K, Recall@K, reciprocal rank, average precision@K, nDCG@K, source precision and recall when reference evidence exists |
| Answer | LLM judge completeness, relevance, correctness, groundedness, missing facts, unsupported claims |
| Citation | Deterministic validity of numbered citations against retrieved evidence |
| Similarity | Lexical similarity and local-embedding cosine similarity against the supplied reference |
| Behavior | Judge abstention score for unanswerable questions; unavailable for answerable questions |
| Operations | Retrieval, answer generation, grading, and total latency |
| Usage and cost | Input, cached input, cache creation, output, total tokens, and estimated USD cost where pricing is known |

An answer passes when completeness and relevance are each at least `0.80`,
preserving the existing pass rule. Additional metrics do not change that rule.
Optional judge metrics are `null` if absent from a judge response; semantic
similarity is `null` with a warning if the embedding model is unavailable.
The dashboard keeps provider comparisons and category charts and exposes the
reference answer, generated answer, metric scores, and warnings for each case.

Legacy evidence/corpus fingerprints still report `current`, `stale`, or
`unverifiable` freshness. Freshness is reported alongside scores; review stale
references manually after corpus changes. User-authored records without freshness
hashes normally report `unverifiable` and remain evaluable. Explicit `null` for
optional lists, evidence, metadata, retrieval configuration, category, or review
status uses the same default as an omitted field. Required strings and malformed
evidence entries still fail validation. Missing evidence yields unavailable
retrieval metrics and a warning, while answer and judge scoring continue. Empty
retrieval results are passed to the grounded prompt so the model can abstain.

## Jobs

`POST /api/v1/evaluations/run/async` returns a job ID and status URL. Polling
reports `queued`, `running`, `completed`, `failed`, `deferred`, `scheduled`,
`stopped`, or `canceled`, plus completed/total model-answer counts and the current
dataset ID. The result is returned inside `result` when the job completes. Individual and
aggregate runs carry `status` (`completed`, `partial`, or `failed`). If every case
errors, the job reports `failed` and retains its results for inspection. A valid
low score remains a scored failure, distinct from an evaluation exception.

Every failed case stores `error_category`, `failed_stage`, a sanitized `error`,
`error_provider`, `error_model`, dataset/record IDs, and run/job IDs. Expand a case
to see these diagnostics. Earlier persisted results have their original messages
but cannot recover stage/model metadata that was never saved. Backend logs retain
exception types and traceback frames; upstream message bodies, headers, source
lines, and local values are omitted because providers can echo API keys, including
partially masked keys. Failed cases have no fabricated scores. All stages run
inside the question boundary; the error builder does not recalculate metrics.

Provider requests disable SDK retries and use at most three attempts for transient
connection/timeouts, server errors, or rate limits with short bounded backoff.
Authentication, permissions, invalid models, and exhausted quota are not retried.
Redis retains results for 24 hours by default; preserve important reports before
expiry. Persist Redis storage to retain job state across service restarts.

LLM grading and estimated cost have practical uncertainty. Use representative,
reviewed reference files and inspect failures before making quality decisions.


## Diagnosis verified on 2026-11-02

The saved OpenAI-answer jobs `02fca7b9-b93a-4b12-a4c0-19204472796a`,
`e4ea5c13-2728-43c9-88e8-439e6bbf90c6`, and
`27d3aef3-9391-45d3-a061-4a1bf966a992` each contain 36 cases with the same saved
OpenAI credential rejection and no generated answer. A minimal live request from
the worker independently returned HTTP **401**, code **invalid_api_key** for the
configured OpenAI model. The selected Grok judge independently returned HTTP
**400** with an incorrect-key rejection. No secret values were retained in this
report. Both services could read and decrypt both credentials; ingestion was
completed and one worker was registered. The uploaded `EmployeeData` dataset
validated all 36 schema-v2 records, including two without reference evidence.

The rejection occurs during answer-provider invocation, before the judge or
score calculation. The old UI replaced any case error with the generic
“Evaluation error” row label. The backend also treated all-error runs as completed
and did not store stage/category metadata. These behaviors concealed a shared
provider failure as many independent question failures.

Replace the rejected OpenAI and Grok credentials in Settings. Valid model access,
account permissions, quota, and a successful live evaluation remain unverified
until accepted credentials are available. Passing mocked tests does not establish
a successful live evaluation.

Regression tests: `python -m pytest -q`, then `npm test`, `npm run lint`,
`npx tsc -b`, and `npm run build` in `frontend/`. The optional
`tests/integration_evaluation_redis.py` smoke test uses a unique real Redis queue,
a temporary JSONL upload, a real RQ worker and result persistence, and mocked
retrieval/model responses. It cleans up only its own test queue and job.
