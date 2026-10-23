"""CLI entry point for Redis Queue evaluation workers."""

from rq import Worker
from rq.serializers import JSONSerializer

from app.core.redis_client import get_evaluation_queue, get_redis_connection


def main() -> None:
    queue = get_evaluation_queue()
    worker = Worker(
        [queue],
        connection=get_redis_connection(),
        serializer=JSONSerializer,
    )
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
