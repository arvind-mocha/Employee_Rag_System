"""Real Redis atomic budget tests; pipe to Python in the API container."""
import asyncio
from contextlib import suppress
from uuid import uuid4

from app.core.redis_client import get_redis_connection
from app.evaluation.rate_limits import ProviderLimits, RedisProviderGovernor, _ACQUIRE


def main():
    connection = get_redis_connection()
    namespace = f"evaluation-budget-test:{uuid4()}"
    limits = {p: ProviderLimits(concurrency=2, answer_concurrency=1, judge_concurrency=2,
                                requests_per_minute=600_000, tokens_per_minute=100_000)
              for p in ('openai','grok','gemini','anthropic')}
    a = RedisProviderGovernor(connection,limits,namespace)
    b = RedisProviderGovernor(connection,limits,namespace)

    async def run():
        active = maximum = answers = max_answers = 0
        async def request(governor,role):
            nonlocal active,maximum,answers,max_answers
            async with governor.slot('openai',role,100,1) as usage:
                active += 1
                maximum = max(maximum,active)
                if role == 'answer':
                    answers += 1
                    max_answers = max(max_answers,answers)
                await asyncio.sleep(.04)
                if role == 'answer': answers -= 1
                active -= 1
                usage['tokens'] = 10
        await asyncio.gather(*(request(a if i%2 else b,'answer' if i%3 else 'judge') for i in range(12)))
        assert maximum == 2 and max_answers == 1
        assert connection.zcard(f'{namespace}:openai:active') == 0
        assert int(connection.hget(f'{namespace}:openai:amounts','used')) == 120
        # Cooldown is visible to a different governor/job instance.
        await a.cooldown('grok',.2)
        allowed, wait = await b._eval(_ACQUIRE,'grok','probe','judge',2,2,600_000,100_000,100,30_000)
        assert not allowed and wait > 0
        # RPM and TPM are independent rolling budgets, shared across roles.
        for suffix, rpm, tpm, tokens in [('rpm',1,100_000,1),('tpm',600_000,150,100)]:
            local = RedisProviderGovernor(connection,limits,namespace+':'+suffix)
            first = await local._eval(_ACQUIRE,'gemini','one','answer',8,8,rpm,tpm,tokens,30_000)
            assert first[0] == 1
            await asyncio.sleep(.002)
            blocked = await local._eval(_ACQUIRE,'gemini','two','judge',8,8,rpm,tpm,tokens,30_000)
            assert blocked[0] == 0 and blocked[1] > 50_000
        # Async cancellation releases admission leases before returning.
        entered = asyncio.Event()
        async def canceled():
            async with a.slot('anthropic','judge',100,1):
                entered.set()
                await asyncio.sleep(10)
        task = asyncio.create_task(canceled())
        await entered.wait()
        task.cancel()
        with suppress(asyncio.CancelledError): await task
        assert connection.zcard(f'{namespace}:anthropic:active') == 0
        assert connection.zcard(f'{namespace}:anthropic:judge') == 0
        # Expired leases from interrupted workers do not permanently occupy caps.
        key=f'{namespace}:grok:active'
        connection.zadd(key,{'dead-worker':0})
        connection.delete(f'{namespace}:grok:cooldown')
        async with a.slot('grok','judge',100,1):
            assert connection.zscore(key,'dead-worker') is None
        print('PASS: shared cross-job/role caps, rolling RPM/TPM, token reconciliation, shared cooldown, cancellation cleanup, expired worker leases')
    try:
        asyncio.run(run())
    finally:
        keys=list(connection.scan_iter(f'{namespace}:*'))
        if keys: connection.delete(*keys)


if __name__=='__main__':
    main()
