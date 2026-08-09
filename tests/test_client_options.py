"""Tests for request timeout and retry configuration."""

from __future__ import annotations

from explainable_agent.agent import ExplainableAgent
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient


def _settings(tmp_path, **overrides) -> Settings:
    return Settings.from_env().with_overrides(
        requested_model="test-model",
        runs_dir=tmp_path / "runs",
        workspace_root=tmp_path,
        **overrides,
    )


def test_settings_expose_timeout_and_retry_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("AGENT_MAX_RETRIES", raising=False)

    settings = Settings.from_env()

    assert settings.request_timeout == 120.0
    assert settings.max_retries == 2


def test_timeout_and_retries_are_configurable_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_REQUEST_TIMEOUT", "33")
    monkeypatch.setenv("AGENT_MAX_RETRIES", "5")

    settings = Settings.from_env()

    assert settings.request_timeout == 33.0
    assert settings.max_retries == 5


def test_agent_passes_timeout_and_retries_to_the_sdk_client(tmp_path):
    """Settings must reach the SDK, not just sit in the dataclass."""
    settings = _settings(tmp_path, request_timeout=7.5, max_retries=0)

    agent = ExplainableAgent(settings=settings)

    assert agent.client.client.timeout == 7.5
    assert agent.client.client.max_retries == 0


def test_client_without_explicit_options_keeps_sdk_defaults():
    """Omitting the arguments must not override the SDK's own defaults."""
    client = OpenAICompatClient(base_url="http://localhost:1234/v1", api_key="k")

    assert client.client.max_retries == 2
    assert client.client.timeout is not None
