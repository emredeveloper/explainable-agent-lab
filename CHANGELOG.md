# Changelog

## 0.3.1 - 2026-08-09

### Fixed

- **Security: `sqlite_execute` guard bypass.** The tool validated only the first
  SQL token but executed the script with `executescript()`, so a chained
  statement such as `INSERT ...; DROP TABLE customers;` ran unchecked. Every
  statement is now validated against the CREATE/INSERT/UPDATE/DELETE allowlist,
  using a splitter that ignores semicolons inside string literals and comments.
- **First-step heuristics hijacked ordinary prose.** Dates, version numbers and
  ranges were parsed as arithmetic — "Summarize the report for Q1-2026" was
  silently answered with `-2025.0` — and English sentences containing SQL verbs
  ("select the best option") were routed to SQLite. Math detection now requires
  an operator between two operands plus a valid AST parse, and SQL detection
  requires real clause structure (`SELECT ... FROM`, `INSERT INTO`, ...).
- **File-path detection matched version strings.** "version 3.10-rc1" resolved to
  the path `3.10`; a candidate now needs a directory separator or a known file
  extension.
- **Report writing crashed on mixed timestamps.** `_duration_ms` raised
  `TypeError` when one timestamp was timezone-naive; both are now normalized to
  UTC before subtraction.
- **`.env` values kept their quotes.** `OPENAI_API_KEY="key"` was sent to the
  server including the quote characters. Values are now unquoted and stripped,
  `export FOO=bar` lines are supported, and invalid numeric values raise an error
  naming the offending variable instead of a bare `ValueError`.
- **CI could not run tests on a fresh checkout.** `pytest --basetemp=.tmp/pytest`
  failed because the git-ignored `.tmp/` parent did not exist.

### Removed

- Dead `_messages_to_prompt` helper in `orchestrator.py` and an unreachable
  ATTACH/DETACH branch in `sqlite_execute`.

### Internal

- Translated the remaining Turkish comments, console output and eval report
  strings to English; annotated the Turkish stopword sets that are functional
  data rather than prose.
- Added `tests/test_bugfix_regressions.py` covering each fix above.

## 0.3.0 - 2026-06-17

- Promoted the package API for library use: exported trace dataclasses, tool helpers, tool registry, and artifact writers from `explainable_agent`.
- Added `py.typed` packaging metadata so type checkers can treat the package as typed.
- Added `ToolRegistry` for isolated per-agent tool catalogs while keeping the existing global `@define_tool` decorator backward compatible.
- Hardened SQLite tools: read queries use a read-only connection with an authorizer, `PRAGMA` is no longer routed through `sqlite_query`, and `sqlite_execute` rejects destructive schema commands outside CREATE/INSERT/UPDATE/DELETE. (**Note:** the `sqlite_execute` guard checked only the first statement in a script and could be bypassed by chaining; fixed in 0.3.1.)
- Tightened deterministic explicit tool calls so tool names only bypass the model when used in `tool_name:` or `/tool_name:` command form.
- Improved orchestrator planning with schema-backed JSON retrieval, normalized delegation plans, and explicit diagnostics when no sub-agent tasks can run.
- Reused the shared relaxed JSON parser inside the OpenAI-compatible client instead of maintaining duplicate parser logic.
- Removed hard-coded SQL table guidance from orchestration reports and fixed the HF tool-calling evaluation default dataset path.
- Expanded regression tests for SQLite guards, explicit-tool routing, orchestrator planning, default eval data, public exports, and isolated tool registries.

## 0.2.5 - 2026-05-03

- Updated release metadata, README install guidance, and PyPI-facing project description for the current local-first Ollama workflow.
- Removed committed local `.env` values from the repository and kept `.env.example` as the only environment template.
- Added GitHub Actions CI with compile, ruff lint/format, pytest, and package build checks.
- Added focused smoke/unit tests for deterministic tool calls, environment loading, path safety, and tool faithfulness scoring.
- Normalized Python formatting with ruff across package, examples, and scripts.
- Refreshed examples to use `.env`/CLI settings instead of hard-coded LM Studio model names, with fast default showcase behavior and optional slower scenarios.
- Improved tool support scoring so numeric and path-like tool outputs are credited correctly in faithfulness diagnostics.

## 0.2.0 - 2026-04-01

- Added OpenAI-native tool calling mode, streaming decision support, and per-step token accounting across traces and reports.
- Refactored the single-agent runtime to separate message building, decision requests, heuristic correction, step recording, and tool follow-up prompts for easier maintenance.
- Improved provider compatibility by using safer plain decision calls for Ollama-compatible servers when structured/native modes are likely to be slow or fragile.
- Updated validation and prepublish flows to rely on lightweight compile checks and runnable scripts.
- Switched DuckDuckGo dependency from `duckduckgo-search` to `ddgs`.

## 0.1.4 - 2026-02-24

- Added a **developer-focused verbose mode**: when `verbose=True`, the agent now prints an \"Agent tools flow roadmap\", per-step decision source/latency, and a compact run summary panel.
- Introduced a **concise summary mode** for `verbose=False`: prints only step counts, a one-line tool flow, final answer preview, and key warnings.
- Validated the library with **LM Studio** using `zai-org/glm-4.6v-flash` and local JSONL benchmarks under `data/evals/` (HF tool-calling and SWE-bench Lite).
- Updated documentation and test helper script to reflect release `v0.1.4`.

## 0.1.0 - 2026-02-17

- Packaged project with `pyproject.toml` and console entrypoint `explainable-agent`.
- Improved README with clear library-style quickstart and evaluation workflows.
- Added MIT license and repository hygiene updates (`.gitignore`).
- Refactored CLI settings override flow via `Settings.with_overrides`.
- Removed `web_search` tool and related heuristics/tests/dataset for a cleaner core scope.
