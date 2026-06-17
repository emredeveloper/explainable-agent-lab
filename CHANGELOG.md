# Changelog

## 0.3.0 - 2026-06-17

- Promoted the package API for library use: exported trace dataclasses, tool helpers, tool registry, and artifact writers from `explainable_agent`.
- Added `py.typed` packaging metadata so type checkers can treat the package as typed.
- Added `ToolRegistry` for isolated per-agent tool catalogs while keeping the existing global `@define_tool` decorator backward compatible.
- Hardened SQLite tools: read queries use a read-only connection with an authorizer, `PRAGMA` is no longer routed through `sqlite_query`, and `sqlite_execute` rejects destructive schema commands outside CREATE/INSERT/UPDATE/DELETE.
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
