"""Tests for selecting the safe RQ worker implementation."""

from unittest.mock import patch

from rq import SpawnWorker, Worker

from app.api.worker import get_worker_class


def test_uses_spawn_worker_on_macos() -> None:
    with patch("app.api.worker.sys.platform", "darwin"):
        assert get_worker_class() is SpawnWorker


def test_uses_standard_worker_on_linux() -> None:
    with patch("app.api.worker.sys.platform", "linux"):
        assert get_worker_class() is Worker
