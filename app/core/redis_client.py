"""Shared Redis connection and RQ queue configuration."""

from dataclasses import dataclass
from functools import lru_cache
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv
from redis import Redis
from rq import Queue
from rq.serializers import JSONSerializer


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_EVALUATION_QUEUE = "employee-rag-evaluations"
DEFAULT_JOB_TIMEOUT_SECONDS = 30 * 60
DEFAULT_RESULT_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class RedisSettings:
    url: str = DEFAULT_REDIS_URL
    evaluation_queue: str = DEFAULT_EVALUATION_QUEUE
    job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT_SECONDS
    result_ttl_seconds: int = DEFAULT_RESULT_TTL_SECONDS

    @classmethod
    def from_env(cls) -> "RedisSettings":
        load_dotenv()
        url = os.getenv("REDIS_URL", DEFAULT_REDIS_URL).strip()
        queue_name = os.getenv(
            "REDIS_EVALUATION_QUEUE",
            DEFAULT_EVALUATION_QUEUE,
        ).strip()
        settings = cls(
            url=url or DEFAULT_REDIS_URL,
            evaluation_queue=queue_name or DEFAULT_EVALUATION_QUEUE,
            job_timeout_seconds=_positive_int_env(
                "REDIS_JOB_TIMEOUT_SECONDS",
                DEFAULT_JOB_TIMEOUT_SECONDS,
            ),
            result_ttl_seconds=_positive_int_env(
                "REDIS_RESULT_TTL_SECONDS",
                DEFAULT_RESULT_TTL_SECONDS,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"redis", "rediss", "unix"}:
            raise ValueError(
                "REDIS_URL must use redis://, rediss://, or unix://"
            )


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


@lru_cache(maxsize=1)
def get_redis_settings() -> RedisSettings:
    return RedisSettings.from_env()


@lru_cache(maxsize=1)
def get_redis_connection() -> Redis:
    """Return one pooled binary Redis client shared by API integrations."""
    return Redis.from_url(
        get_redis_settings().url,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )


def get_evaluation_queue() -> Queue:
    settings = get_redis_settings()
    return Queue(
        name=settings.evaluation_queue,
        connection=get_redis_connection(),
        serializer=JSONSerializer,
        default_timeout=settings.job_timeout_seconds,
    )
