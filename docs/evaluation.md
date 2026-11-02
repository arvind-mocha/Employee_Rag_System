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

## JSONL schema

Use UTF-8 (an optional UTF-8 BOM is accepted), with one JSON object per nonempty
line. Blank lines are ignored. Every record requires two nonblank strings:

```jsonl
{"question":"How many vacation days do employees receive?","reference_answer":"Employees receive 15 vacation days annually."}
{"id":"unknown-office-code","question":"What is the office access code?","reference_answer":"The documents do not provide this information.","answerable":false}
```

These sample facts are illustrative; supply references that match your corpus.
A fuller example is available at [examples/evaluation.jsonl](../examples/evaluation.jsonl).

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
hashes normally report `unverifiable` and remain evaluable.

## Jobs

`POST /api/v1/evaluations/run/async` returns a job ID and status URL. Polling
reports `queued`, `running`, `completed`, `failed`, `deferred`, `scheduled`,
`stopped`, or `canceled`, plus completed/total model-answer counts and the current
dataset ID. The result is returned inside `result` when the job completes.
Redis retains results for 24 hours by default; preserve important reports before
expiry. Persist Redis storage to retain job state across service restarts.

LLM grading and estimated cost have practical uncertainty. Use representative,
reviewed reference files and inspect failures before making quality decisions.
