"""Atomic, shared evaluation budgets. No provider requests proceed if Redis fails."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import random
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

from app.api import config
from app.core.redis_client import get_redis_connection


class ProviderLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    concurrency: int = Field(default=4, ge=1, le=8)
    answer_concurrency: int = Field(default=4, ge=1, le=8)
    judge_concurrency: int = Field(default=4, ge=1, le=8)
    requests_per_minute: int = Field(default=30, ge=1)
    tokens_per_minute: int = Field(default=60_000, ge=1)


def configured_limits() -> dict[str, ProviderLimits]:
    overrides = json.loads(config.EVALUATION_PROVIDER_LIMITS)
    if not isinstance(overrides, dict) or set(overrides) - {"openai", "gemini", "anthropic", "grok"}:
        raise ValueError("EVALUATION_PROVIDER_LIMITS must map supported provider names to limits")
    return {p: ProviderLimits.model_validate(overrides.get(p, {}))
            for p in ("openai", "gemini", "anthropic", "grok")}


# One Redis script handles leases, role caps, a rolling RPM/TPM window, smooth
# request pacing, and a shared cooldown. Redis time avoids clock skew across jobs.
_ACQUIRE = """
local base, id, role = KEYS[1], ARGV[1], ARGV[2]
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
local cap, rolecap, rpm, tpm, tokens, lease = unpack(ARGV, 3)
cap, rolecap, rpm, tpm, tokens, lease = tonumber(cap), tonumber(rolecap), tonumber(rpm), tonumber(tpm), tonumber(tokens), tonumber(lease)
local active, roles, requests, window, amounts = base..':active', base..':'..role, base..':requests', base..':tokens', base..':amounts'
redis.call('ZREMRANGEBYSCORE', active, '-inf', now)
redis.call('ZREMRANGEBYSCORE', roles, '-inf', now)
redis.call('ZREMRANGEBYSCORE', requests, '-inf', now-60000)
local expired = redis.call('ZRANGEBYSCORE', window, '-inf', now-60000)
local used = tonumber(redis.call('HGET', amounts, 'used') or '0')
for _, old in ipairs(expired) do
    used = used-tonumber(redis.call('HGET', amounts, old) or '0')
    redis.call('HDEL', amounts, old)
end
redis.call('ZREMRANGEBYSCORE', window, '-inf', now-60000)
used = math.max(0, used)
redis.call('HSET', amounts, 'used', used)
local cooldown = tonumber(redis.call('GET', base..':cooldown') or '0')
local next = tonumber(redis.call('GET', base..':next') or '0')
if cooldown > now or next > now then return {0, math.max(cooldown, next)-now} end
if redis.call('ZCARD', active) >= cap or redis.call('ZCARD', roles) >= rolecap then return {0, 100} end
if redis.call('ZCARD', requests) >= rpm then
    local first = redis.call('ZRANGE', requests, 0, 0, 'WITHSCORES')
    return {0, tonumber(first[2])+60001-now}
end
if used+tokens > tpm then
    local first = redis.call('ZRANGE', window, 0, 0, 'WITHSCORES')
    return {0, tonumber(first[2])+60001-now}
end
redis.call('ZADD', active, now+lease, id)
redis.call('ZADD', roles, now+lease, id)
redis.call('ZADD', requests, now, id)
redis.call('ZADD', window, now, id)
redis.call('HSET', amounts, id, tokens, 'used', used+tokens)
redis.call('SET', base..':next', now+math.ceil(60000/rpm), 'PX', 120000)
for _, key in ipairs({active, roles, requests, window, amounts}) do redis.call('PEXPIRE', key, math.max(120000, lease+1000)) end
return {1, 0}
"""
_RELEASE = """
redis.call('ZREM', KEYS[1]..':active', ARGV[1])
redis.call('ZREM', KEYS[1]..':'..ARGV[2], ARGV[1])
-- Reconcile reservations only while they are still in the rolling window.
local old = redis.call('HGET', KEYS[1]..':amounts', ARGV[1])
if old and tonumber(ARGV[3]) >= 0 then
    local used = tonumber(redis.call('HGET', KEYS[1]..':amounts', 'used') or '0')
    redis.call('HSET', KEYS[1]..':amounts', ARGV[1], ARGV[3], 'used', math.max(0, used-tonumber(old)+tonumber(ARGV[3])))
end
return 1
"""
_COOLDOWN = """
local clock = redis.call('TIME')
local until_ms = tonumber(clock[1])*1000+math.floor(tonumber(clock[2])/1000)+tonumber(ARGV[1])
local existing = tonumber(redis.call('GET', KEYS[1]..':cooldown') or '0')
if until_ms > existing then redis.call('SET', KEYS[1]..':cooldown', until_ms, 'PX', tonumber(ARGV[1])+1000) end
return 1
"""


class ProviderBudgetError(RuntimeError):
    """A fixed, secret-free error for unusable evaluation budgets."""


class ProviderBudgetUnavailable(ProviderBudgetError):
    """Shared budget storage is unavailable; do not proceed without limits."""


class RedisProviderGovernor:
    def __init__(self, connection=None, limits=None, namespace="employee-rag:evaluation:limits:v1"):
        self.connection = connection
        self.limits = limits if limits is not None else configured_limits()
        self.namespace = namespace

    async def _eval(self, script, provider, *args):
        try:
            connection = self.connection or get_redis_connection()
            task = asyncio.create_task(asyncio.to_thread(connection.eval, script, 1, f"{self.namespace}:{provider}", *args))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Finish a pending atomic reservation before finally releasing it.
                await task
                raise
        except (RedisError, OSError) as error:
            raise ProviderBudgetUnavailable("Evaluation rate-limit storage is unavailable. Restore Redis and retry.") from error

    @asynccontextmanager
    async def slot(self, provider, role, tokens, timeout_seconds):
        limits = self.limits[provider]
        if tokens > limits.tokens_per_minute:
            raise ProviderBudgetError(
                f"The {provider} request exceeds its configured evaluation token budget. Increase tokens_per_minute or reduce retrieved context."
            )
        request_id = str(uuid4())
        usage = {"tokens": -1}  # Failed/unknown usage retains its conservative reservation.
        try:
            while True:
                allowed, wait_ms = await self._eval(
                    _ACQUIRE, provider, request_id, role, limits.concurrency,
                    limits.answer_concurrency if role == "answer" else limits.judge_concurrency,
                    limits.requests_per_minute, limits.tokens_per_minute, tokens,
                    (timeout_seconds + 30) * 1000,
                )
                if allowed:
                    break
                await asyncio.sleep(min(float(wait_ms) / 1000, 1) + random.uniform(0, .05))
            yield usage
        finally:
            await asyncio.shield(self._eval(_RELEASE, provider, request_id, role, usage["tokens"]))

    async def cooldown(self, provider, delay):
        await self._eval(_COOLDOWN, provider, max(1, int(delay * 1000)))


def retry_delay(error, attempt: int) -> float:
    """Honor seconds or HTTP-date Retry-After; never retry earlier than requested."""
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    minimum = 0.0
    if value:
        try:
            minimum = max(0, float(value))
        except (ValueError, TypeError):
            try:
                minimum = max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return max(minimum, min(30, .5 * (2 ** attempt))) + random.uniform(0, .25)
