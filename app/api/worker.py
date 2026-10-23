"""CLI entry point for Redis Queue evaluation workers."""

import sys

from rq import SpawnWorker, Worker
from rq.serializers import JSONSerializer

from app.core.redis_client import get_evaluation_queue, get_redis_connection


def get_worker_class() -> type[Worker]:
    """Use a spawn-based worker where macOS makes fork unsafe for PyTorch."""
    if sys.platform == "darwin":
        return SpawnWorker
    return Worker


def main() -> None:
    queue = get_evaluation_queue()
    worker = get_worker_class()(
        [queue],
        connection=get_redis_connection(),
        serializer=JSONSerializer,
    )
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
