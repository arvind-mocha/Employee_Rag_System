"""RQ checkpoints: immutable case writes and coordinator-owned progress snapshots."""
from datetime import datetime, timezone
from hashlib import sha256

from app.core.redis_client import get_redis_settings
from app.evaluation.engine import _summary, performance_summary
from app.evaluation.models import EvaluationCaseResult, EvaluationRun, EvaluationBatchRun


def case_key(job_id):
    return f"employee-rag:evaluation:{job_id}:cases"


def cancellation_key(job_id):
    return f"employee-rag:evaluation:{job_id}:cancel"


class EvaluationJobStore:
    def __init__(self, job):
        self.job = job
        self.ttl = get_redis_settings().result_ttl_seconds

    def manifest(self, datasets, run_id):
        self.job.meta.update(checkpoint_version=1, dataset_manifest=datasets, run_id=run_id)
        self.job.save_meta()

    def persist(self, result):
        # All case IDs include run, dataset, provider and record identity. HSETNX
        # makes a repeated completion harmless without overwriting earlier data.
        with self.job.connection.pipeline(transaction=True) as pipe:
            pipe.hsetnx(case_key(self.job.id), result.case_id, result.model_dump_json())
            pipe.expire(case_key(self.job.id), self.ttl)
            pipe.execute()

    def progress(self, snapshot):
        # Only the coordinator writes meta, so completed/progress fields cannot
        # race with worker threads. Cancellation uses a separate Redis key.
        self.job.meta.update(snapshot)
        self.job.save_meta()

    def should_cancel(self):
        return bool(self.job.connection.exists(cancellation_key(self.job.id)))


def checkpoint_response(job, metadata, status, elapsed_seconds):
    """Recover saved cases after cancellation, timeout, or worker termination."""
    request = metadata.get("evaluation_request")
    if metadata.get("checkpoint_version") != 1 or not isinstance(request, dict):
        return None
    from app.api.llm import LLMProvider
    cases = [EvaluationCaseResult.model_validate_json(value)
             for value in job.connection.hvals(case_key(job.id))]
    cases = [case for case in cases if case.run_id == metadata["run_id"]]
    cases.sort(key=lambda r: (r.dataset_order, r.record_order, r.provider_order))
    providers = [LLMProvider(p) for p in request["answer_providers"]]
    judge = LLMProvider(request["judge_provider"])
    run_id = metadata["run_id"]
    manifests = metadata.get("dataset_manifest", [])
    error = "Evaluation interrupted; saved case results are preserved." if status in {"failed", "stopped"} else "Evaluation canceled; completed results are preserved." if status == "canceled" else None

    def make_run(name, dataset_id, digest, subset):
        return EvaluationRun(
            run_id=run_id, dataset_id=dataset_id, dataset_name=name, dataset_sha256=digest,
            generated_at=job.started_at or job.enqueued_at or datetime.now(timezone.utc),
            judge_provider=judge, concurrency=request.get("concurrency", 4),
            status="canceled" if status == "canceled" else "failed" if status in {"failed", "stopped"} else "partial",
            error=error, results=subset, summaries=[_summary(p, subset) for p in providers],
            performance=performance_summary(subset, elapsed_seconds, metadata.get("maximum_concurrent_cases", 0)),
        )
    datasets = [make_run(m["filename"], m["dataset_id"], m["dataset_sha256"],
                         [r for r in cases if r.dataset_id == m["dataset_id"]]) for m in manifests]
    aggregate_hash = sha256("\n".join(f"{m['dataset_id']}:{m['dataset_sha256']}" for m in manifests).encode()).hexdigest()
    aggregate = make_run(f"{len(manifests)} selected datasets", None, aggregate_hash, cases)
    # The partial snapshot uses the same usage calculation as final reports.
    from app.api.llm_usage import summarize_llm_usage
    for run in [*datasets, aggregate]:
        run.usage = summarize_llm_usage(u for r in run.results for u in (r.answer_usage, r.judge_usage))
    return EvaluationBatchRun(run_id=run_id, datasets=datasets, aggregate=aggregate).model_dump(mode="json")
