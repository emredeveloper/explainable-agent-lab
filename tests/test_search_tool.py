"""Tests for the web search tool's network bounds."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import explainable_agent.tools as tools


def _install_fake_ddgs(monkeypatch, recorder: dict, *, accept_timeout: bool = True):
    class FakeDDGS:
        def __init__(self, *_args, **kwargs):
            if not accept_timeout and "timeout" in kwargs:
                raise TypeError("unexpected keyword argument 'timeout'")
            recorder["timeout"] = kwargs.get("timeout")

        def text(self, _query, max_results=None):
            recorder["max_results"] = max_results
            return [{"title": "t", "href": "https://example.com", "body": "b"}]

    module = types.ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)


def test_search_passes_a_timeout_so_it_cannot_hang(monkeypatch):
    recorder: dict = {}
    _install_fake_ddgs(monkeypatch, recorder)

    result = tools.duckduckgo_search("python", Path.cwd())

    assert recorder["timeout"] == tools.SEARCH_TIMEOUT_SECONDS
    assert recorder["max_results"] == tools.SEARCH_MAX_RESULTS
    assert "example.com" in result


def test_search_falls_back_when_client_rejects_timeout(monkeypatch):
    """Older duckduckgo-search builds do not accept the keyword."""
    recorder: dict = {}
    _install_fake_ddgs(monkeypatch, recorder, accept_timeout=False)

    result = tools.duckduckgo_search("python", Path.cwd())

    assert not result.startswith("ERROR:")
    assert "example.com" in result


def test_search_rejects_empty_query():
    assert tools.duckduckgo_search("   ", Path.cwd()).startswith("ERROR:")
