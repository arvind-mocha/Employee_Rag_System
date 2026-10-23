"""Shared Redis connection and RQ queue configuration."""

from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit

from redis import Redis
from rq import Queue
from rq.serializers import JSONSerializer

from app.api import config


DEFAULT_REDIS_URL = config.REDIS_URL
DEFAULT_EVALUATION_QUEUE = config.REDIS_EVALUATION_QUEUE
DEFAULT_JOB_TIMEOUT_SECONDS = config.REDIS_JOB_TIMEOUT_SECONDS
DEFAULT_RESULT_TTL_SECONDS = config.REDIS_RESULT_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class RedisSettings:
    url: str = DEFAULT_REDIS_URL
    evaluation_queue: str = DEFAULT_EVALUATION_QUEUE
    job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT_SECONDS
    result_ttl_seconds: int = DEFAULT_RESULT_TTL_SECONDS

    @classmethod
    def from_config(cls) -> "RedisSettings":
        settings = cls()
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"redis", "rediss", "unix"}:
            raise ValueError(
                "REDIS_URL must use redis://, rediss://, or unix://"
            )


@lru_cache(maxsize=1)
def get_redis_settings() -> RedisSettings:
    return RedisSettings.from_config()


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
