# Changelog

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
