# RAG quality evaluation

The evaluation subsystem measures the retrieval and answer-generation pipeline
against a versioned JSONL ground-truth dataset. It is explicitly triggered and
never runs as a side effect of ingestion, retrieval, or normal chat.

## Prerequisites

Before evaluating:

1. Complete document ingestion.
2. Configure every provider that will generate references, answer questions, or
   judge results.
3. Start Redis and the evaluation worker:

```bash
uv run employee-rag-worker
```

4. Decide which retrieval configuration should be tested.

The frontend uses asynchronous jobs so that long-running model calls do not
block an HTTP request.

## Create a dataset

There are two supported generation workflows. The customer-facing frontend
generates the JSONL internally instead of asking users to prepare or upload it.

### Generate from the frontend

1. Open **Evaluation** in Developer mode.
2. Select dataset generation.
3. Enter a dataset name and choose one to five questions per category.
4. Select the reference provider.
5. Select the retrieval strategies.
6. Queue generation and wait for the worker.

To remove a stored dataset, enter or select its name on the Evaluation page,
choose the trash action beside the dataset field, and confirm the deletion.
Deleting evaluation ground truth does not remove ingested source documents.

The worker reuses the cleaned documents and structured-data artifacts retained
by ingestion. It selects source-diverse excerpts and asks the reference provider
to generate balanced questions across all nine categories in bounded batches.
The generator identifies the source excerpts supporting each answerable question;
those anchors are preserved when retrieval evidence is added, so a retrieval miss
does not incorrectly turn an answerable case into an unanswerable one. The worker
then produces an answerability decision, expected keywords, reference answer,
required facts, and relevant evidence positions.
Generation starts only when **Generate ground truth** is selected. One to five
questions per category produces 9 to 45 records.

### Generate from the command line

Create a questions file containing one JSON object per line:

```jsonl
{"question":"How many vacation days do employees receive?"}
{"question":"What is the remote work policy?"}
```

An example is included at `evaluation_questions.example.jsonl`.

Run:

```bash
uv run employee-rag-generate-eval \
  --dataset employee-policies \
  --questions evaluation_questions.example.jsonl \
  --reference-provider openai \
  --reranking
```

Optional Boolean flags include semantic search, keyword search, query rewriting,
reranking, and source diversity. Each flag also accepts its `--no-...` form.
Use `--overwrite` only when replacing a dataset intentionally.

## Ground-truth records

Each record contains:

- Stable record ID and schema version
- Question, answerability, evaluation category, and expected keywords
- Reference answer and atomic required facts
- Retrieved and relevant evidence snapshots
- Chunk IDs, sources, sections, positions, and content hashes
- Evidence and corpus fingerprints
- Reference provider, model, generation time, and usage
- Retrieval configuration used during generation
- Review status

The reference model assigns one primary category to every generated record:
`direct_fact`, `temporal`, `spanning`, `relationship`, `numerical`,
`comparative`, `holistic`, `procedural`, or `unanswerable`. These stored
categories are carried into evaluation results and drive the category
performance chart; the frontend does not infer them from question wording.

Generated records start with `review_status: machine_generated`. After verifying
the question, answer, facts, and evidence, change the status to `approved`. Do
not treat a machine-generated reference as authoritative without review.

## Freshness checking

Evidence and corpus fingerprints connect an evaluation record to the indexed
data from which it was generated.

| Status | Meaning |
| --- | --- |
| `current` | Its evidence still matches the current corpus |
| `stale` | Relevant evidence or corpus content has changed |
| `unverifiable` | The current system cannot prove freshness |

Stale records have `passed: null` and are excluded from pass-rate aggregates.
Regenerate and review the dataset after material source-data changes.

## Run an evaluation

From the frontend:

1. Select a stored dataset.
2. Select one to four answer providers.
3. Select a judge provider.
4. Choose retrieval strategies.
5. Start the run and wait for its job result.

Each answer provider receives the same question and retrieval configuration. The
judge provider assesses qualities that require semantic judgment, while
deterministic code calculates retrieval, citation, similarity, timing, and
usage metrics.

## Metrics

| Category | Metrics |
| --- | --- |
| Retrieval | Precision@K, Recall@K, reciprocal rank, average precision@K, nDCG@K, source precision, source recall |
| Answer | Correctness, groundedness, completeness, relevance, overall score |
| Citation | Citation accuracy, citation coverage, deterministic citation validity |
| Behavior | Abstention correctness, unsupported claims, missing facts |
| Similarity | Lexical similarity and local-embedding semantic similarity |
| Operations | Retrieval, answer, grading, and total latency |
| Usage | Input, cached input, cache creation, output, and total tokens |
| Cost | Per-call and aggregate approximate USD cost when model pricing is available |

An answer passes when overall, correctness, and groundedness are each at least
`0.80`.

Provider summaries report evaluated, passed, failed, errored, stale, and
unreviewed counts, plus averages, pass rate, latency, usage, and cost.

## Job lifecycle

Asynchronous requests return a job ID and status URL. Possible states include
`queued`, `running`, `completed`, `failed`, `deferred`, `scheduled`, `stopped`,
and `canceled`.

Successful jobs return the normal operation result inside `result`. Failed jobs
return a safe error. Redis retains job success and failure results for 24 hours
by default, so record important results before they expire.

The frontend stores the latest accepted dataset-generation and evaluation job
IDs locally. Navigating to another page, using chat, or closing the Evaluation
page does not cancel a queued or running worker job. Opening Evaluation again
restores the job, resumes status polling, and loads its result from Redis. The
browser is only observing the job; execution belongs to the RQ worker.

## Limitations

- LLM judges can be inconsistent or biased toward particular styles or model
  families.
- Generated ground truth can contain errors.
- Approximate cost may differ from provider invoices.
- A small question set may not represent actual employee usage.
- High retrieval scores do not guarantee policy correctness.
- Evaluation should support, not replace, expert and user review.

Use diverse, approved questions that cover normal, ambiguous, unanswerable, and
high-risk cases.
