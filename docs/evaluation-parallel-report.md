# Parallel RAG evaluation implementation

## Implementation

Default case concurrency is **4**, validated from **1–8**. The UI offers
1/2/3/4/6/8 and retains the selection. `EVALUATION_CONCURRENCY` sets the default
for API clients omitting the request field. Existing RQ workers and all scoring
prompts, score definitions, pass thresholds, usage, and cost calculations remain.

The bounded async coordinator schedules at most the chosen number of case tasks
and threads. Threads execute the existing synchronous case pipeline; provider
requests bridge to native LangChain `ainvoke` on one event loop. A waiting judge
does not block other cases. This preserves one shared scoring implementation
instead of creating divergent synchronous/async metric paths. Concurrency one
retains sequential execution. No new dependencies or multiprocessing were added.

## Provider controls and resource safety

Atomic Redis scripts share provider-wide concurrency, role concurrency, rolling
RPM/TPM, paced admissions, and cooldowns across evaluation jobs and preflight.
Defaults for each provider are 4 total/answer/judge requests, 30 RPM, and 60,000
TPM. Configure actual known account limits through `EVALUATION_PROVIDER_LIMITS`;
answer/judge role limits remain separate while their shared provider's quotas
are combined. Detailed configuration is in [evaluation.md](evaluation.md#parallel-evaluation).

Token reservations estimate input conservatively and include the maximum output
allowance; usage reconciles them afterward. HTTP 429 remains authoritative.
Retries use bounded exponential backoff with jitter and `Retry-After`, at most
three attempts. Authentication, permission, quota exhaustion, and unsupported
models receive no retry. A shared cooldown is published even on an exhausted
attempt to protect other jobs. Missing Redis never permits unlimited requests.

Local retrieval and embedding initialization/inference run under a process lock,
protecting synchronous PGVector wrappers and local model state while network
waits overlap. Credentials use independent short-lived psycopg connections;
provider clients belong to individual invocations and close their async HTTP
transports. No database session or mutable case state is shared.

One coordinator persists immutable case hashes with `HSETNX` and updates RQ
progress. IDs include run/dataset/provider/record identity. Stored order indices
restore dataset/record/provider order even when cases finish out of order.
Retrieved context remains attached to its own case. Progress includes total,
completed, active, scored, error, and low-score counts, percentage, elapsed wall
time, maximum active cases, and retry counts.

Cancellation stops new scheduling and cancels native provider I/O, including
preflight. Synchronous retrieval/inference finishes safely. Saved cases survive
cancellation or interrupted worker states and are reconstructed by the status
endpoint. Admission leases expire after a dead worker's request timeout plus a
safety margin. No automatic resumption of unfinished cases is added.

## Measured benchmark

Synthetic external I/O only: approximately 50 ms per answer and 50 ms per judge,
plus a small controlled retrieval delay. The real prompts, native async path,
judge parsing, metrics, ordering, and aggregation execute. The benchmark replaces
provider budgets with a no-wait test governor to measure scheduler scaling;
separate real Redis tests verify the budgets. No paid provider calls were made.

| Cases | Concurrency | Wall time (s) | Average retrieval (ms) | Answer (ms) | Judge (ms) | Case (ms) | Peak active | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.124 | 1.35 | 59.97 | 61.79 | 123.20 | 1 | — |
| 4 | 4 | 0.123 | 3.20 | 53.44 | 55.27 | 111.98 | 4 | — |
| 36 | 1 | 3.892 | 1.30 | 52.14 | 53.89 | 107.45 | 1 | 1.00× |
| 36 | 4 | 0.989 | 1.87 | 52.45 | 54.22 | 108.65 | 4 | **3.94×** |
| 36 | 8 | 0.546 | 2.76 | 52.01 | 53.64 | 108.51 | 8 | **7.13×** |

Every benchmark case scored successfully, with zero execution errors and retries.
[Raw benchmark data](evaluation-parallel-benchmark.json) retains all measurements.
Run `PYTHONPATH=.:tests .venv/bin/python tests/benchmark_evaluation_parallel.py`
from the project root to reproduce. Timings naturally vary across machines.

## Verification

- Backend: **246 passed**, plus **28 passed subtests**.
- Frontend: **16 passed**.
- ESLint, TypeScript `tsc -b`, and Vite production build: passed.
- Real Redis limits integration: shared caps across independent governor instances,
  separate role limits, rolling RPM/TPM, actual-usage reconciliation, cooldowns,
  cancellation cleanup, and expired worker leases passed.
- Real Redis/RQ integration: four cases, immediate immutable checkpoints,
  duplicate-write protection, real worker execution, persisted HTTP results,
  simulated hard-interruption recovery, and cancellation signaling passed.
- Mocked native-I/O tests: exactly-once normal execution, deterministic ordering,
  independent per-case client ownership, matching answers/context/metrics,
  sequential/parallel score consistency, one failed judge, progress accuracy,
  multiple datasets, cancellation before/during preflight and active requests,
  persistence failure, and all four providers' bounded retries passed.
- Compose resolved API and worker settings to the same defaults: concurrency 4
  and an empty override object.

Live provider speedup was not measured; the earlier 20-minute run is a historical
baseline. Five minutes at concurrency four remains a target, subject to provider
latency, actual quotas, shared account usage, and local processing.

## Adoption and limitations

Existing provider credentials and `.env` contents were not changed. The existing
Docker services were not redeployed during verification. To adopt source/image
changes and newly configured environment variables, rebuild/recreate the local
services after any active evaluation finishes:

```sh
docker compose up -d --build api worker frontend
```

The UI's eight active cases do not override the default provider cap of four;
raise provider caps only when actual account limits support that workload.
Token estimates cannot precisely predict provider-specific token accounting or
requests from other applications. The budgets coordinate this installation's
evaluations, not unrelated chat traffic or external clients. Selected datasets
run in selection order, with parallel cases within each dataset. Local retrieval
and inference remain serialized. A hard worker stop cannot run cleanup; leases
expire and saved results remain available. Checkpoints retain the existing
24-hour TTL. Unfinished cases are not automatically resumed.

## Files modified or added

- `app/api/config.py`
- `app/api/controllers/evaluation.py`
- `app/api/controllers/evaluation_jobs.py`
- `app/api/llm.py`
- `app/api/routes/post.py`
- `app/api/schemas/evaluation.py`
- `app/api/schemas/jobs.py`
- `app/core/evaluation_context.py`
- `app/evaluation/engine.py`
- `app/evaluation/jobs.py`
- `app/evaluation/models.py`
- `app/evaluation/persistence.py`
- `app/evaluation/preflight.py`
- `app/evaluation/rate_limits.py`
- `docker-compose.yml`
- `docs/evaluation-parallel-benchmark.json`
- `docs/evaluation-parallel-report.md`
- `docs/evaluation.md`
- `frontend/src/components/EvaluationProgress.tsx`
- `frontend/src/components/ParallelEvaluationSettings.tsx`
- `frontend/src/lib/api.ts`
- `frontend/src/lib/evaluationConcurrency.ts`
- `frontend/src/pages/EvaluationPage.tsx`
- `frontend/src/styles.css`
- `frontend/src/types/api.ts`
- `frontend/tests/evaluationResults.test.mjs`
- `tests/benchmark_evaluation_parallel.py`
- `tests/integration_evaluation_limits.py`
- `tests/integration_evaluation_redis.py`
- `tests/test_evaluation_api.py`
- `tests/test_parallel_evaluation.py`
