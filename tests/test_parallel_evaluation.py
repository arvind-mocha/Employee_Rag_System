"""Deterministic native-I/O concurrency, routing, cancellation, and retry tests."""
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import re
import threading
from time import perf_counter, sleep
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.api.llm import ChatModelClient, LLMProvider, LLMProviderRequestError, ainvoke_chat_model
from app.api.schemas.evaluation import RunEvaluationRequest
from app.evaluation.dataset import text_sha256
from app.evaluation.engine import run_evaluation
from app.evaluation.models import EvidenceSnapshot, GroundTruthRecord, JudgeDraft, RetrievalConfiguration
from app.evaluation.rate_limits import ProviderLimits, ProviderBudgetError, RedisProviderGovernor, retry_delay
from app.retrieval.models import RetrievedChunk


class TestGovernor:
    __test__ = False
    def __init__(self):
        self.calls = []
        self.cooldowns = []

    @asynccontextmanager
    async def slot(self, provider, role, tokens, timeout):
        self.calls.append((provider, role, tokens))
        yield {"tokens": -1}

    async def cooldown(self, provider, delay):
        self.cooldowns.append((provider, delay))


class Harness:
    """Mock only external I/O; use real prompts, parsing, scoring and aggregation."""
    def __init__(self, failure=None, delay=.025):
        self.failure = failure
        self.delay = delay
        self.active = self.maximum = 0
        self.answers = []
        self.judges = []
        self.retrievals = []
        self.client_ids = []

    def records(self, count):
        return [GroundTruthRecord(id=f"r{i}", question=f"q{i}", reference_answer=f"answer q{i}",
                relevant_evidence=[EvidenceSnapshot(content=f"answer q{i}", content_sha256=text_sha256(f"answer q{i}"))]) for i in range(count)]

    def search(self, question, options):
        self.retrievals.append(question)
        sleep(.001)
        return [RetrievedChunk(content=f"answer {question}", relevance_score=1, chunk_id=question,
                               source_file=f"{question}.md", source_type="markdown", section=None, chunk_index=0)]

    def client(self, provider, **kwargs):
        harness = self
        class Model:
            structured = False
            def with_structured_output(self, *args, **kwargs):
                self.structured = True
                return self
            async def ainvoke(self, messages):
                harness.active += 1
                harness.maximum = max(harness.maximum, harness.active)
                try:
                    if self.structured:
                        payload = json.loads(messages[-1].content.split('<evaluation_input>\n')[1].split('\n</evaluation_input>')[0])
                        question = payload['question']
                        harness.judges.append(question)
                        # Verify answers never cross between cases.
                        assert payload['candidate_answer'] == f"answer {question} [1]"
                    else:
                        question = re.search(r'Question: (q\d+)', messages[-1].content).group(1)
                        harness.answers.append(question)
                    # Force completion order to differ from dataset order.
                    await asyncio.sleep(harness.delay + (.009 if question == 'q0' else 0))
                    if self.structured and question == harness.failure:
                        raise RuntimeError('private upstream detail')
                    metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                    if self.structured:
                        return {"raw": AIMessage(content="", usage_metadata=metadata), "parsed": JudgeDraft(
                            completeness=1, relevance=1, correctness=1, groundedness=1, rationale=f"Supported {question}"), "parsing_error": None}
                    return AIMessage(content=f"answer {question} [1]", usage_metadata=metadata)
                finally:
                    harness.active -= 1
        model = Model()
        self.client_ids.append(model)
        return ChatModelClient(provider, "mock-model", model)

    def run(self, count, concurrency, **kwargs):
        with patch('app.api.llm.get_chat_model_client', side_effect=self.client), \
             patch('app.evaluation.engine.semantic_similarity', return_value=1):
            return run_evaluation('dataset', self.records(count), [LLMProvider.OPENAI], LLMProvider.GROK,
                                  RetrievalConfiguration(), concurrency=concurrency,
                                  router=Mock(search=self.search), content_sha256='hash', governor=TestGovernor(), **kwargs)


@pytest.mark.parametrize('count,concurrency', [(1,4),(4,4),(36,4),(36,8),(4,1)])
def test_native_parallel_pipeline_exactly_once_order_and_bounded_requests(count, concurrency):
    harness = Harness()
    persisted, snapshots, threads = [], [], []
    def save(result):
        persisted.append(result)
        threads.append(threading.get_ident())
    result = harness.run(count, concurrency, on_case_result=save, on_progress=snapshots.append)
    assert result.status == 'completed'
    assert [r.record_id for r in result.results] == [f'r{i}' for i in range(count)]
    assert len({r.case_id for r in result.results}) == count
    assert Counter(harness.answers) == Counter(f'q{i}' for i in range(count))
    assert Counter(harness.judges) == Counter(harness.answers)
    assert len(harness.client_ids) == 2 * count
    assert harness.maximum <= concurrency
    assert result.performance.maximum_concurrent_cases == min(count, concurrency)
    assert set(threads) == {threading.get_ident()}  # All writes on coordinator.
    assert len(persisted) == count
    if count > 1 and concurrency > 1:
        assert [r.record_id for r in persisted] != [r.record_id for r in result.results]
    for r in result.results:
        assert r.answer == f'answer {r.question} [1]'
        assert r.retrieved_context == [f'answer {r.question}']
        assert r.retrieval_scores.recall_at_k == 1
        assert r.answer_scores.correctness == 1
        assert r.usage.total_tokens == 30
        assert r.dataset_id == 'dataset'
    assert result.usage.total_tokens == 30 * count
    assert [s['completed_cases'] for s in snapshots] == sorted(s['completed_cases'] for s in snapshots)
    assert all(s['running_cases'] <= concurrency for s in snapshots)
    assert snapshots[-1]['completed_cases'] == count and snapshots[-1]['running_cases'] == 0
    assert snapshots[-1]['successful_cases'] == count and snapshots[-1]['failed_cases'] == 0
    assert snapshots[-1]['progress_percent'] == 100


def test_parallel_speedup_and_identical_metrics():
    sequential = Harness().run(36, 1)
    parallel = Harness().run(36, 4)
    assert parallel.performance.elapsed_seconds < sequential.performance.elapsed_seconds * .65
    for left, right in zip(sequential.results, parallel.results):
        assert left.answer_scores == right.answer_scores
        assert left.retrieval_scores == right.retrieval_scores
        assert left.usage == right.usage
    assert parallel.performance.elapsed_seconds < sum(r.timing.total_ms for r in parallel.results) / 1000


def test_one_native_judge_failure_does_not_stop_other_cases():
    result = Harness(failure='q2').run(8, 4)
    assert result.status == 'partial'
    assert len(result.results) == 8
    assert result.performance.successful_cases == 7 and result.performance.error_cases == 1
    assert result.results[2].failed_stage == 'judge_invocation'
    assert result.results[2].answer_scores is None
    assert 'private upstream' not in result.results[2].error


def test_cancellation_stops_scheduling_cleans_up_native_requests_and_preserves_results():
    saved = []
    event = threading.Event()
    def save(result):
        saved.append(result)
        event.set()
    harness = Harness(delay=.05)
    result = harness.run(36, 4, on_case_result=save, should_cancel=event.is_set)
    assert result.status == 'canceled'
    assert 1 <= len(result.results) <= 4
    assert len(harness.answers) <= 4
    assert harness.active == 0
    assert result.results == sorted(saved, key=lambda r:r.record_order)


def test_mid_request_cancellation_cancels_native_io():
    harness = Harness(delay=.5)
    event = threading.Event()
    timer = threading.Timer(.05, event.set)
    timer.start()
    try:
        started = perf_counter()
        result = harness.run(36, 4, should_cancel=event.is_set)
    finally:
        timer.cancel()
    assert result.status == 'canceled'
    assert result.results == []
    assert harness.active == 0
    assert perf_counter()-started < .5
    assert len(harness.answers) <= 4


def test_persistence_failure_is_not_discarded_or_reported_completed():
    harness = Harness()
    with pytest.raises(RuntimeError, match='persistence unavailable'):
        harness.run(36, 4, on_case_result=Mock(side_effect=RuntimeError('persistence unavailable')))
    assert len(harness.answers) <= 4
    assert harness.active == 0


@pytest.mark.parametrize('value', [0,9,-1,True,2.5,'4',None])
def test_concurrency_request_validation(value):
    with pytest.raises(ValidationError):
        RunEvaluationRequest(dataset_ids=['data'], answer_providers=['openai'], judge_provider='grok', concurrency=value)


@pytest.mark.parametrize('value', [1,2,3,4,5,6,7,8])
def test_api_accepts_every_integer_in_configuration_range(value):
    assert RunEvaluationRequest(dataset_ids=['data'], answer_providers=['openai'], judge_provider='grok', concurrency=value).concurrency == value


class UpstreamError(Exception):
    def __init__(self, status, code=None, retry_after=None):
        super().__init__('private secret')
        self.status_code, self.code = status, code
        self.response = Mock(headers={'Retry-After': retry_after} if retry_after else {})


@pytest.mark.parametrize('provider', list(LLMProvider))
def test_every_provider_uses_role_budgets_and_bounded_429_retries(provider):
    from unittest.mock import AsyncMock
    model = Mock(ainvoke=AsyncMock(side_effect=[UpstreamError(429), AIMessage(content='OK')]))
    governor = TestGovernor()
    with patch('app.api.llm.get_chat_model_client', return_value=ChatModelClient(provider,'model',model)), \
         patch('app.evaluation.rate_limits.retry_delay', return_value=0):
        result = asyncio.run(ainvoke_chat_model([HumanMessage(content='Q')],provider,governor=governor))
    assert result.text == 'OK'
    assert model.ainvoke.call_count == 2
    assert len(governor.cooldowns) == 1
    assert all(call[:2] == (provider.value, 'answer') for call in governor.calls)


@pytest.mark.parametrize('status,code', [(401,'invalid_api_key'),(404,'model_not_found'),(429,'insufficient_quota')])
def test_native_permanent_failures_are_not_retried(status,code):
    from unittest.mock import AsyncMock
    model = Mock(ainvoke=AsyncMock(side_effect=UpstreamError(status,code)))
    with patch('app.api.llm.get_chat_model_client', return_value=ChatModelClient(LLMProvider.OPENAI,'model',model)):
        with pytest.raises(LLMProviderRequestError):
            asyncio.run(ainvoke_chat_model([],LLMProvider.OPENAI,governor=TestGovernor()))
    assert model.ainvoke.call_count == 1


def test_retry_after_seconds_date_and_jitter():
    from email.utils import format_datetime
    from datetime import timedelta
    with patch('app.evaluation.rate_limits.random.uniform', return_value=.1):
        assert retry_delay(UpstreamError(429,retry_after='12'),0) == 12.1
        future = format_datetime(datetime.now(timezone.utc)+timedelta(seconds=30))
        assert 28 < retry_delay(UpstreamError(429,retry_after=future),0) < 31
        assert retry_delay(UpstreamError(429,retry_after='bad'),2) == 2.1


def test_no_retry_if_retry_after_exceeds_bounded_wait():
    from unittest.mock import AsyncMock
    model = Mock(ainvoke=AsyncMock(side_effect=UpstreamError(429,retry_after='600')))
    governor = TestGovernor()
    with patch('app.api.llm.get_chat_model_client', return_value=ChatModelClient(LLMProvider.OPENAI,'model',model)):
        with pytest.raises(LLMProviderRequestError):
            asyncio.run(ainvoke_chat_model([],LLMProvider.OPENAI,governor=governor))
    assert model.ainvoke.call_count == 1
    assert governor.cooldowns[0][1] >= 600


def test_unknown_quota_defaults_are_bounded_and_oversized_request_is_rejected():
    governor = RedisProviderGovernor()
    assert all(limit.requests_per_minute == 30 for limit in governor.limits.values())
    assert all(limit.tokens_per_minute == 60_000 for limit in governor.limits.values())
    async def run():
        with pytest.raises(ProviderBudgetError, match='token budget'):
            async with governor.slot('openai','judge',60_001,120):
                pytest.fail('oversized request was admitted')
    asyncio.run(run())


def test_multiple_datasets_share_global_progress_and_keep_grouping():
    from app.api.controllers.evaluation import execute_selected_evaluation
    harness = Harness()
    snapshots, persisted = [], []
    metadata = [dict(dataset_id=f'd{i}', filename='same.jsonl', dataset_sha256=f'hash{i}') for i in range(2)]
    selected = [(m,harness.records(4)) for m in metadata]
    request = RunEvaluationRequest(dataset_ids=['d0','d1'],answer_providers=['openai'],judge_provider='grok',concurrency=4)
    with patch('app.api.controllers.evaluation.load_selected_datasets',return_value=selected), \
         patch('app.api.controllers.evaluation.validate_evaluation_providers'), \
         patch('app.api.controllers.evaluation.validate_runtime_preflight',return_value={}), \
         patch('app.evaluation.engine.RedisProviderGovernor',side_effect=TestGovernor), \
         patch('app.evaluation.engine.get_query_router',return_value=Mock(search=harness.search)), \
         patch('app.api.llm.get_chat_model_client',side_effect=harness.client), \
         patch('app.evaluation.engine.semantic_similarity',return_value=1):
        result = execute_selected_evaluation(request,on_case_result=persisted.append,on_progress=snapshots.append,job_id='job')
    assert len(result.aggregate.results) == 8
    assert len({r.case_id for r in persisted}) == 8
    assert [r.dataset_id for r in result.aggregate.results] == ['d0']*4+['d1']*4
    assert [r.record_order for r in result.aggregate.results] == list(range(4))*2
    assert all(r.job_id == 'job' and r.run_id == result.run_id for r in persisted)
    assert snapshots[-1]['completed_cases'] == 8
    assert snapshots[-1]['successful_cases'] == 8
    assert snapshots[-1]['progress_percent'] == 100
    assert all(s['total_cases'] == 8 for s in snapshots)
    assert [s['completed_cases'] for s in snapshots] == sorted(s['completed_cases'] for s in snapshots)
    assert result.aggregate.usage.total_tokens == 240


def test_retry_counts_propagate_from_native_requests_to_case_and_run():
    class RetryHarness(Harness):
        failed_once = False
        def client(self,provider,**kwargs):
            client = super().client(provider,**kwargs)
            original = client.model.ainvoke
            async def invoke(messages):
                if provider is LLMProvider.OPENAI and not self.failed_once:
                    self.failed_once = True
                    raise UpstreamError(429)
                return await original(messages)
            client.model.ainvoke = invoke
            return client
    with patch('app.evaluation.rate_limits.retry_delay',return_value=0):
        result = RetryHarness().run(4,4)
    assert result.performance.retry_count == 1
    assert sum(r.retry_count for r in result.results) == 1
    assert result.performance.successful_cases == 4


def test_native_retries_exhaust_after_three_attempts():
    from unittest.mock import AsyncMock
    model = Mock(ainvoke=AsyncMock(side_effect=UpstreamError(429)))
    governor = TestGovernor()
    with patch('app.api.llm.get_chat_model_client',return_value=ChatModelClient(LLMProvider.OPENAI,'model',model)), \
         patch('app.evaluation.rate_limits.retry_delay',return_value=0):
        with pytest.raises(LLMProviderRequestError):
            asyncio.run(ainvoke_chat_model([],LLMProvider.OPENAI,governor=governor))
    assert model.ainvoke.call_count == 3
    assert len(governor.cooldowns) == 3


def test_native_clients_are_closed_on_success_and_cancellation():
    from unittest.mock import AsyncMock
    class Model:
        def __init__(self):
            self.root_async_client = Mock(close=AsyncMock())
        async def ainvoke(self,messages):
            await asyncio.sleep(.1)
            return AIMessage(content='OK')
    model = Model()
    with patch('app.api.llm.get_chat_model_client',return_value=ChatModelClient(LLMProvider.OPENAI,'model',model)):
        asyncio.run(ainvoke_chat_model([],LLMProvider.OPENAI,governor=TestGovernor()))
    model.root_async_client.close.assert_awaited_once()
    model.root_async_client.close.reset_mock()
    async def cancel():
        task = asyncio.create_task(ainvoke_chat_model([],LLMProvider.OPENAI,governor=TestGovernor()))
        await asyncio.sleep(.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    with patch('app.api.llm.get_chat_model_client',return_value=ChatModelClient(LLMProvider.OPENAI,'model',model)):
        asyncio.run(cancel())
    model.root_async_client.close.assert_awaited_once()


def test_budget_storage_failure_does_not_fall_back_to_unlimited_requests():
    from unittest.mock import AsyncMock
    from app.evaluation.rate_limits import ProviderBudgetUnavailable
    class UnavailableGovernor(TestGovernor):
        @asynccontextmanager
        async def slot(self,*args):
            raise ProviderBudgetUnavailable('Evaluation rate-limit storage is unavailable. Restore Redis and retry.')
            yield
    model = Mock(ainvoke=AsyncMock())
    with patch('app.api.llm.get_chat_model_client',return_value=ChatModelClient(LLMProvider.OPENAI,'model',model)):
        with pytest.raises(LLMProviderRequestError) as failure:
            asyncio.run(ainvoke_chat_model([],LLMProvider.OPENAI,governor=UnavailableGovernor()))
    assert failure.value.category == 'provider_unavailable'
    assert failure.value.retryable
    model.ainvoke.assert_not_called()


@pytest.mark.parametrize('state,expected', [('queued','canceled'),('started','requested'),('finished','already_finished')])
def test_cancel_endpoint_supports_rq_job_lifecycle(state,expected):
    from fastapi.testclient import TestClient
    from app.api.main import app
    job = Mock(id='job',func_name='app.evaluation.jobs.run_evaluation_job')
    job.get_status.return_value = state
    with patch('app.api.controllers.evaluation_jobs.Job.fetch',return_value=job), \
         patch('app.api.controllers.evaluation_jobs.get_redis_connection'), \
         patch('app.api.controllers.evaluation_jobs.get_evaluation_queue'):
        response = TestClient(app).post('/api/v1/evaluations/jobs/job/cancel')
    assert response.status_code == 202
    assert response.json()['status'] == expected
    assert job.cancel.call_count == (1 if state=='queued' else 0)
    assert job.connection.set.call_count == (0 if state=='finished' else 1)


def test_invalid_provider_limit_configuration_is_blocked_before_any_provider_call():
    from fastapi import HTTPException
    from app.evaluation.preflight import validate_runtime_preflight
    with patch('app.api.config.EVALUATION_PROVIDER_LIMITS','{"openai":{"concurrency":9}}'), \
         patch('app.evaluation.preflight.invoke_chat_model') as invoke:
        with pytest.raises(HTTPException,match='EVALUATION_PROVIDER_LIMITS'):
            validate_runtime_preflight([LLMProvider.OPENAI],LLMProvider.GROK)
    invoke.assert_not_called()


def test_application_concurrency_setting_supplies_request_default():
    with patch('app.api.config.EVALUATION_CONCURRENCY',3):
        request = RunEvaluationRequest(dataset_ids=['d'],answer_providers=['openai'],judge_provider='grok')
    assert request.concurrency == 3


def test_request_budgets_are_provider_specific():
    from app.evaluation.rate_limits import configured_limits
    with patch('app.api.config.EVALUATION_PROVIDER_LIMITS','{"openai":{"concurrency":2,"requests_per_minute":10,"tokens_per_minute":12345},"grok":{"judge_concurrency":3}}'):
        limits = configured_limits()
    assert limits['openai'].concurrency == 2 and limits['openai'].requests_per_minute == 10
    assert limits['openai'].tokens_per_minute == 12345
    assert limits['grok'].judge_concurrency == 3
    assert limits['anthropic'].concurrency == 4 and limits['gemini'].concurrency == 4


def test_cancellation_before_preflight_does_not_invoke_providers():
    from app.core.evaluation_context import EvaluationCancelled
    from app.evaluation.preflight import validate_runtime_preflight
    with patch('app.evaluation.preflight.invoke_chat_model') as invoke:
        with pytest.raises(EvaluationCancelled):
            validate_runtime_preflight([LLMProvider.OPENAI],LLMProvider.GROK,should_cancel=lambda:True)
    invoke.assert_not_called()


def test_cancellation_during_preflight_native_request():
    from app.core.evaluation_context import EvaluationCancelled
    from app.evaluation.preflight import validate_runtime_preflight
    from unittest.mock import AsyncMock
    event = threading.Event()
    async def slow(messages):
        await asyncio.sleep(5)
        return AIMessage(content='OK')
    model = Mock(ainvoke=AsyncMock(side_effect=slow))
    timer = threading.Timer(.05,event.set)
    timer.start()
    try:
        with patch('app.evaluation.preflight.ingestion_runner') as ingestion, \
             patch('app.evaluation.preflight.RedisProviderGovernor',return_value=TestGovernor()), \
             patch('app.api.llm.get_chat_model_client',return_value=ChatModelClient(LLMProvider.OPENAI,'model',model)):
            ingestion.get_status.return_value.status = 'completed'
            start = perf_counter()
            with pytest.raises(EvaluationCancelled):
                validate_runtime_preflight([LLMProvider.OPENAI],LLMProvider.GROK,should_cancel=event.is_set)
            assert perf_counter()-start < 1
    finally:
        timer.cancel()
