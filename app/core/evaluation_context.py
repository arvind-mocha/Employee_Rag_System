"""Evaluation-only async bridge; ordinary chat requests keep their existing path."""
import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Event


class EvaluationCancelled(Exception):
    """Cooperative cancellation, never converted into a fake scored case."""


@dataclass
class CaseTrace:
    retries: int = 0


EVALUATION_RUNTIME: ContextVar = ContextVar("evaluation_runtime", default=None)
CASE_TRACE: ContextVar = ContextVar("evaluation_case_trace", default=None)


class EvaluationRuntime:
    def __init__(self, governor):
        self.loop = asyncio.get_running_loop()
        self.governor = governor
        self.cancelled = Event()
        self.pending = set()
        self.tasks = set()

    def invoke(self, messages, provider, **kwargs):
        from app.api.llm import ainvoke_chat_model
        if self.cancelled.is_set():
            raise EvaluationCancelled()
        async def request():
            task = asyncio.current_task()
            self.tasks.add(task)
            try:
                if self.cancelled.is_set():
                    raise asyncio.CancelledError()
                return await ainvoke_chat_model(messages, provider, governor=self.governor, **kwargs)
            finally:
                self.tasks.discard(task)

        future = asyncio.run_coroutine_threadsafe(request(), self.loop)
        # Threads only wait for their own async request. No provider client is
        # shared, and only the loop mutates pending via call_soon_threadsafe.
        self.loop.call_soon_threadsafe(self.pending.add, future)
        try:
            return future.result()
        except (asyncio.CancelledError, FutureCancelledError):
            raise EvaluationCancelled() from None
        finally:
            self.loop.call_soon_threadsafe(self.pending.discard, future)

    def cancel(self):
        if self.cancelled.is_set():
            return
        self.cancelled.set()
        for future in tuple(self.pending):
            future.cancel()
