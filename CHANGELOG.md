# Changelog

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
