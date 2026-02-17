# Architecture

## Core modules

- `explainable_agent/agent.py`
  - Main orchestration loop
  - Decision -> tool call -> next decision flow
  - Trace/audit generation
- `explainable_agent/lmstudio_client.py`
  - LM Studio OpenAI-compatible client wrapper
  - Structured decision parsing
- `explainable_agent/tools.py`
  - Built-in tools and guards (file/sql/math)
- `explainable_agent/report.py`
  - `trace.json`, `trace_full.json`, `report.md` artifacts
- `explainable_agent/eval_tool_calls.py`
  - Tool-call normalization and scoring logic

## Runtime flow

1. User task enters via CLI.
2. Agent asks model for a structured next action.
3. If action is `tool_call`, selected tool runs with guarded input.
4. Tool output is fed back into the next model step.
5. Agent exits on `final_answer` or max-step cutoff.
6. Artifacts are written to `runs/<run_id>/`.

## Evaluation flow

1. Load dataset rows (`jsonl` or BFCL SQL).
2. Prompt model to output tool calls.
3. Parse + repair + guard predicted calls.
4. Score against expected calls.
5. Write summary/details/report under `runs/evals/`.
