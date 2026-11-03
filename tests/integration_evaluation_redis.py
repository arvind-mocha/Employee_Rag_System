"""Opt-in smoke test against a running Redis, with mocked model/retrieval I/O.

Run from the project root with PYTHONPATH=. python
tests/integration_evaluation_redis.py, or pipe it to Python in the API container. Uses a unique queue, temporary datasets, and deletes only its own job.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from rq import Queue, SimpleWorker
from rq.serializers import JSONSerializer

from app.api.llm import ChatModelClient, LLMProvider
from app.api.main import app
from app.core.redis_client import get_redis_connection
from app.evaluation.models import JudgeDraft
from app.retrieval.models import RetrievedChunk


def main():
    connection = get_redis_connection()
    queue = Queue(f"evaluation-regression-{uuid4()}", connection=connection, serializer=JSONSerializer)
    client = TestClient(app)
    answer, judge = Mock(), Mock()
    answer.invoke.return_value = AIMessage(content="15 days annually [1].")
    judge.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": JudgeDraft(completeness=1, relevance=1, rationale="Supported."),
        "parsing_error": None,
    }
    clients = {LLMProvider.OPENAI: ChatModelClient(LLMProvider.OPENAI, "mock-answer", answer),
               LLMProvider.GROK: ChatModelClient(LLMProvider.GROK, "mock-judge", judge)}
    chunk = RetrievedChunk(content="15 days annually.", relevance_score=1, chunk_id="policy-1",
                           source_file="policy.md", source_type="markdown", section=None, chunk_index=0)
    job = None
    try:
        with TemporaryDirectory() as directory, \
             patch("app.api.config.EVALUATION_DATA_DIRECTORY", Path(directory)), \
             patch("app.api.controllers.evaluation_jobs.get_evaluation_queue", return_value=queue), \
             patch("app.evaluation.preflight.get_evaluation_queue", return_value=queue), \
             patch("app.evaluation.preflight.Worker.count", return_value=1), \
             patch("app.api.controllers.evaluation.validate_evaluation_providers"), \
             patch("app.api.controllers.evaluation_jobs.validate_evaluation_providers"), \
             patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
             patch("app.api.llm.get_chat_model_client", side_effect=lambda provider, **_: clients[provider]), \
             patch("app.evaluation.engine.get_query_router", return_value=Mock(search=Mock(return_value=[chunk]))), \
             patch("app.evaluation.engine.semantic_similarity", return_value=1):
            ingestion.get_status.return_value.status = "completed"
            uploaded = client.post("/api/v1/evaluations/datasets/upload", files={
                "dataset_file": ("regression.jsonl", json.dumps({
                    "question": "How many vacation days?", "reference_answer": "15 days annually.",
                    "reference_evidence": [{"content": chunk.content}],
                }).encode(), "application/x-ndjson"),
            })
            assert uploaded.status_code == 201, uploaded.text
            dataset_id = uploaded.json()["datasets"][0]["dataset_id"]
            accepted = client.post("/api/v1/evaluations/run/async", json={
                "dataset_ids": [dataset_id], "answer_providers": ["openai"], "judge_provider": "grok",
            })
            assert accepted.status_code == 202, accepted.text
            job_id = accepted.json()["job_id"]
            job = queue.fetch_job(job_id)
            worker = SimpleWorker([queue], connection=connection, serializer=JSONSerializer)
            worker.work(burst=True, with_scheduler=False, logging_level="WARNING")
            # Fetch a fresh Job from Redis through the normal HTTP polling route.
            polled = client.get(f"/api/v1/evaluations/jobs/{job_id}")
            assert polled.status_code == 200, polled.text
            body = polled.json()
            assert body["status"] == "completed", body
            result = body["result"]["aggregate"]["results"][0]
            assert result["answer_scores"]["overall"] == 1
            assert result["retrieval_scores"]["recall_at_k"] == 1
            assert result["job_id"] == job_id and result["dataset_id"] == dataset_id
            assert body["completed_cases"] == 1
            print("PASS: JSONL upload -> real RQ queue/worker -> mocked answer/retrieval/judge -> Redis persistence -> HTTP polling")
    finally:
        if job is not None:
            job.delete()
        queue.delete()


if __name__ == "__main__":
    main()
