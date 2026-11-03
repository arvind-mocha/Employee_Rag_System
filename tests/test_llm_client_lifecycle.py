"""Exercise real SDK transports without external provider requests."""

import asyncio
from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
import httpx2 as httpx
from langchain_core.messages import HumanMessage
import pytest

from app.api.llm import LLMProvider, LLMSettings, ainvoke_chat_model


class Governor:
    @asynccontextmanager
    async def slot(self, *args):
        yield {"tokens": -1}


@pytest.mark.parametrize("provider", [
    LLMProvider.OPENAI, LLMProvider.GROK, LLMProvider.ANTHROPIC,
])
def test_native_clients_survive_repeated_calls_and_event_loops(provider, monkeypatch):
    transports = []

    async def respond(request):
        # Allow other requests to start before any owned transport is closed.
        await asyncio.sleep(.01)
        if provider is LLMProvider.ANTHROPIC:
            payload = {
                "id": "msg_test", "type": "message", "role": "assistant",
                "model": "test-model", "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        else:
            payload = {
                "id": "chat_test", "object": "chat.completion", "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        return httpx.Response(200, json=payload)

    def http_client(**kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        transports.append(client)
        return client

    monkeypatch.setattr("app.api.llm.DefaultAsyncHttpxClient", http_client)
    monkeypatch.setattr(
        "app.api.llm.AsyncAnthropic",
        lambda **kwargs: AsyncAnthropic(**kwargs, http_client=http_client()),
    )
    settings = LLMSettings(
        gemini_api_key="test-key", openai_api_key="test-key", openai_model="test-model",
        grok_api_key="test-key", grok_model="test-model",
        anthropic_api_key="test-key", anthropic_model="test-model",
    )
    monkeypatch.setattr("app.api.llm._stored_llm_settings", lambda _: settings)

    async def invoke():
        return await ainvoke_chat_model(
            [HumanMessage(content="Hello")], provider, governor=Governor(), max_attempts=1,
        )

    async def run():
        assert (await invoke()).text == "Hello"
        assert (await invoke()).text == "Hello"
        results = await asyncio.gather(*(invoke() for _ in range(3)))
        assert [result.text for result in results] == ["Hello"] * 3

    # Preflight and worker runs use separate asyncio.run event loops.
    asyncio.run(run())
    asyncio.run(run())
    assert len(transports) == 10
    assert len({id(client) for client in transports}) == 10
    assert all(client.is_closed for client in transports)


def test_gemini_async_cleanup_closes_the_model(monkeypatch):
    from unittest.mock import AsyncMock, Mock
    from langchain_core.messages import AIMessage
    from app.api.llm import ChatModelClient

    model = Mock(ainvoke=AsyncMock(return_value=AIMessage(content="Hello")), aclose=AsyncMock())
    monkeypatch.setattr(
        "app.api.llm.get_chat_model_client",
        lambda *args, **kwargs: ChatModelClient(LLMProvider.GEMINI, "test-model", model),
    )
    asyncio.run(ainvoke_chat_model([], LLMProvider.GEMINI, governor=Governor()))
    model.aclose.assert_awaited_once()
