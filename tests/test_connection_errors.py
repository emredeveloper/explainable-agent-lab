"""Tests for connection failure reporting.

An unreachable local server is the most common failure mode for this package,
so the error must name the address that was tried and how to change it.
"""

from __future__ import annotations

import pytest
from openai import APIConnectionError

from explainable_agent import LLMConnectionError
from explainable_agent.openai_client import OpenAICompatClient


class _Boom:
    """Stands in for an SDK endpoint whose host is unreachable."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def __call__(self, *_args, **_kwargs):
        raise self.exc

    # models.list() and chat.completions.create() are both plain calls.
    list = create = property(lambda self: self)


def _client_with_dead_server(monkeypatch) -> OpenAICompatClient:
    client = OpenAICompatClient(base_url="http://localhost:59999/v1", api_key="x")
    exc = APIConnectionError(request=None)

    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(client.client.models, "list", _raise)
    monkeypatch.setattr(client.client.chat.completions, "create", _raise)
    return client


def test_list_models_reports_the_address_and_how_to_change_it(monkeypatch):
    client = _client_with_dead_server(monkeypatch)

    with pytest.raises(LLMConnectionError) as excinfo:
        client.list_models()

    message = str(excinfo.value)
    assert "http://localhost:59999/v1" in message
    assert "--base-url" in message


def test_chat_completion_failure_is_reported_as_connection_error(monkeypatch):
    client = _client_with_dead_server(monkeypatch)

    with pytest.raises(LLMConnectionError):
        client.get_decision(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            reasoning_effort="low",
        )


def test_connection_error_is_not_swallowed_by_mode_fallbacks(monkeypatch):
    """Fallbacks exist to downgrade features, not to retry a dead server."""
    client = _client_with_dead_server(monkeypatch)
    calls = {"count": 0}

    original = client._create_chat_completion

    def _counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "_create_chat_completion", _counting)

    with pytest.raises(LLMConnectionError):
        client.get_decision_native(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            reasoning_effort="low",
        )

    # Without the guard the native path would fall back to get_decision and
    # retry against the same unreachable server.
    assert calls["count"] == 1


def test_llm_connection_error_is_a_runtime_error():
    """Existing callers that catch RuntimeError keep working."""
    assert issubclass(LLMConnectionError, RuntimeError)
