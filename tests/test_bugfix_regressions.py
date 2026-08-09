"""Regression tests for previously fixed defects.

Each test pins behaviour that was actually broken, so a future refactor cannot
silently reintroduce the bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from explainable_agent.agent import (
    _extract_math_expression,
    _extract_path_candidate,
    _extract_sql_statement,
    _heuristic_tool_suggestion,
)
from explainable_agent.config import Settings
from explainable_agent.report import _duration_ms
from explainable_agent.tools import run_tool

# --- sqlite_execute chained-statement guard bypass ---------------------------


def test_sqlite_execute_rejects_destructive_statement_after_semicolon(
    tmp_path, monkeypatch
):
    """A chained DROP used to slip past the first-token-only guard."""
    monkeypatch.setenv("AGENT_SQLITE_DB", "data/regression.db")
    assert run_tool("sqlite_init_demo", "", tmp_path).startswith("OK:")

    result = run_tool(
        "sqlite_execute",
        'INSERT INTO customers VALUES (9, "a", "b"); DROP TABLE customers;',
        tmp_path,
    )

    assert result.startswith("ERROR:")
    assert "drop" in result.lower()
    # The table must still be present.
    assert "customers" in run_tool("sqlite_list_tables", "", tmp_path)


def test_sqlite_execute_rejects_attach_in_chained_script(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SQLITE_DB", "data/regression.db")
    assert run_tool("sqlite_init_demo", "", tmp_path).startswith("OK:")

    result = run_tool(
        "sqlite_execute",
        'INSERT INTO customers VALUES (9, "a", "b"); ATTACH DATABASE "x.db" AS x;',
        tmp_path,
    )

    assert result.startswith("ERROR:")


def test_sqlite_execute_still_allows_legitimate_multi_statement_script(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AGENT_SQLITE_DB", "data/regression.db")
    assert run_tool("sqlite_init_demo", "", tmp_path).startswith("OK:")

    result = run_tool(
        "sqlite_execute",
        'INSERT INTO customers VALUES (10, "m", "n"); '
        'UPDATE customers SET city = "z" WHERE id = 10;',
        tmp_path,
    )

    assert result.startswith("OK:")


def test_sqlite_execute_treats_semicolon_inside_string_as_data(tmp_path, monkeypatch):
    """A semicolon inside a literal must not be read as a statement separator."""
    monkeypatch.setenv("AGENT_SQLITE_DB", "data/regression.db")
    assert run_tool("sqlite_init_demo", "", tmp_path).startswith("OK:")

    result = run_tool(
        "sqlite_execute",
        "INSERT INTO customers VALUES (11, 'a;DROP TABLE customers;', 'c')",
        tmp_path,
    )

    assert result.startswith("OK:")
    assert "customers" in run_tool("sqlite_list_tables", "", tmp_path)


# --- heuristic false positives ----------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        "What is the weather in Istanbul on 2026-01-05?",
        "Summarize the report for Q1-2026",
        "Tell me about version 3.10-rc1",
        "Review pages 10-20",
        "Upgrade to release 2.5",
    ],
)
def test_prose_with_numbers_is_not_treated_as_math(task):
    """Dates, versions and ranges used to be evaluated as arithmetic."""
    assert _extract_math_expression(task) is None


@pytest.mark.parametrize(
    "task",
    [
        "Please select the best option and update me later",
        "Can you create a summary of the file?",
        "Delete my account please",
    ],
)
def test_english_prose_using_sql_verbs_is_not_treated_as_sql(task):
    assert _extract_sql_statement(task) is None


@pytest.mark.parametrize(
    "task",
    [
        "What is the weather in Istanbul on 2026-01-05?",
        "Summarize the report for Q1-2026",
        "Please select the best option and update me later",
        "Can you create a summary of the file?",
    ],
)
def test_prose_tasks_produce_no_heuristic_tool_override(task):
    assert _heuristic_tool_suggestion(task) is None


@pytest.mark.parametrize(
    ("task", "expected_tool"),
    [
        ("calculate_math: (215*4)-12", "calculate_math"),
        ("What is (12*7)+3?", "calculate_math"),
        ("compute 100/4", "calculate_math"),
        ("calculate 2026-1999", "calculate_math"),
        ("SELECT name FROM customers", "sqlite_query"),
        ("DELETE FROM orders WHERE id=1", "sqlite_execute"),
        ("list all *.py files", "list_workspace_files"),
        ("read docs/notes.txt", "read_text_file"),
    ],
)
def test_genuine_signals_still_route_to_the_right_tool(task, expected_tool):
    """Tightening the heuristics must not break real tool routing."""
    suggestion = _heuristic_tool_suggestion(task)
    assert suggestion is not None, f"expected a suggestion for {task!r}"
    assert suggestion[0] == expected_tool


def test_version_number_is_not_mistaken_for_a_file_path():
    assert _extract_path_candidate("Tell me about version 3.10-rc1") is None
    assert _extract_path_candidate("read docs/notes.txt") == "docs/notes.txt"


# --- report duration with mixed timezone awareness ---------------------------


def test_duration_ms_handles_mixed_timezone_awareness():
    """Naive/aware mixtures used to raise TypeError while writing artifacts."""
    assert _duration_ms("2026-01-01T00:00:00", "2026-01-01T00:00:02+00:00") == 2000
    assert _duration_ms("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:02") == 2000


def test_duration_ms_returns_zero_for_unparsable_timestamps():
    assert _duration_ms("not-a-date", "also-not-a-date") == 0


# --- .env parsing -------------------------------------------------------------


def test_env_file_values_are_unquoted_and_stripped(tmp_path, monkeypatch):
    """A quoted API key used to be sent to the server including the quotes."""
    monkeypatch.chdir(tmp_path)
    for key in ("OPENAI_API_KEY", "AGENT_MODEL", "AGENT_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)

    Path(".env").write_text(
        "\n".join(
            [
                'OPENAI_API_KEY="quoted-key"',
                "AGENT_MODEL=  spaced-model  ",
                "export AGENT_TEMPERATURE=0.7",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env()

    assert settings.api_key == "quoted-key"
    assert settings.requested_model == "spaced-model"
    assert settings.temperature == 0.7


def test_invalid_numeric_env_value_reports_the_variable_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_MAX_STEPS", "not-a-number")

    with pytest.raises(ValueError, match="AGENT_MAX_STEPS"):
        Settings.from_env()
