# Explainable Agent Lab

Local-first, explainable agent framework for OpenAI-compatible models.

This repository is built for two things:
1. Running a tool-using agent with transparent step-by-step traces.
2. Evaluating tool-calling quality on benchmark-style datasets.

## What You Get

- Structured agent decisions per step:
  - `action`, `confidence`, `rationale`, `evidence`
- Tool execution traces with audit metadata
- Faithfulness signals:
  - alternative answer similarity
  - tool-support score
- Run artifacts:
  - `trace.json` (compact)
  - `trace_full.json` (full payload)
  - `report.md` (human-readable)
- Evaluation pipeline with:
  - dataset adapters (`jsonl`, `bfcl_sql`, `swebench_lite`)
  - random/head sampling
  - parse/repair/guard stats
  - argument-level error breakdown
  - robust JSON parsing/repair for malformed model outputs

## Repository Layout

- `explainable_agent/`: core library
- `scripts/`: evaluation and utility scripts
- `tests/`: unit tests
- `data/evals/`: sample and benchmark datasets
- `docs/ARCHITECTURE.md`: system overview

## Installation

### Option A: Requirements-based setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Editable package install (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Copy `.env.example` and adjust values if needed.

## Quick Start

1. Open your local OpenAI-compatible server.
2. Load a model (e.g. `gpt-oss-20b`).
3. Default endpoint is `http://localhost:1234/v1`.

List available models:

```bash
python -m explainable_agent.cli --list-models
```

Run the agent:

```bash
python -m explainable_agent.cli --model gpt-oss-20b --reasoning-effort high --task "calculate_math: (215*4)-12"
```

Or via console entrypoint:

```bash
explainable-agent --model gpt-oss-20b --task "sqlite_init_demo"
```

## Built-in Tools

- `calculate_math`
- `read_text_file`
- `list_workspace_files`
- `now_utc`
- `sqlite_init_demo`
- `sqlite_list_tables`
- `sqlite_describe_table`
- `sqlite_query`
- `sqlite_execute`

SQLite example:

```bash
python -m explainable_agent.cli --sqlite-db data/demo.db --task "sqlite_init_demo"
python -m explainable_agent.cli --sqlite-db data/demo.db --task "sqlite_query: SELECT name, city FROM customers ORDER BY id;"
```

## Evaluation

Note on output length:
- `--max-completion-tokens` is optional.
- If omitted (or set to `0`), evaluators do not send `max_tokens`; generation ends naturally when the model finishes.

Mini sample run:

```bash
python scripts/eval_hf_tool_calls.py --dataset data/evals/hf_xlam_fc_sample.jsonl --model gpt-oss-20b --reasoning-effort high --limit 10
```

ComplexFuncBench subset:

```bash
python scripts/eval_hf_tool_calls.py --dataset data/evals/hf_complexfuncbench_first_turn_100.jsonl --model gpt-oss-20b --reasoning-effort high --limit 10 --sampling random
```

BFCL SQL:

```bash
python scripts/eval_hf_tool_calls.py --dataset data/evals/bfcl_sql/BFCL_v3_sql.json --model gpt-oss-20b --reasoning-effort high
```

SWE-bench Lite readiness (scalable adapter path):

```bash
python scripts/eval_swebench_readiness.py --dataset data/evals/swebench_lite_test.jsonl --model gpt-oss-20b --limit 10
```

Download real SWE-bench Lite test split:

```bash
python -c "from datasets import load_dataset; ds=load_dataset('SWE-bench/SWE-bench_Lite', split='test'); ds.to_json('data/evals/swebench_lite_test.jsonl', orient='records', lines=True, force_ascii=False)"
```

Outputs are written to:
- `runs/evals/hf_tool_eval_<timestamp>/summary.json`
- `runs/evals/hf_tool_eval_<timestamp>/details.json`
- `runs/evals/hf_tool_eval_<timestamp>/report.md`
- `runs/evals/swebench_readiness_<timestamp>/summary.json`
- `runs/evals/swebench_readiness_<timestamp>/details.json`

## Prepublish Check

Before publishing:

```bash
python scripts/prepublish_check.py
```

## Configuration

- `OPENAI_BASE_URL` (default: `http://localhost:1234/v1`)
- `OPENAI_API_KEY` (default: `local`)
- `AGENT_MODEL` (default: `gpt-oss-20b`)
- `AGENT_REASONING_EFFORT` (default: `high`)
- `AGENT_MAX_STEPS` (default: `6`)
- `AGENT_RUNS_DIR` (default: `runs`)
- `AGENT_WORKSPACE` (default: `.`)
- `AGENT_TEMPERATURE` (default: `0.2`)
- `AGENT_SQLITE_DB` (default: `data/agent.db`)

## Status

Current release: `v0.1.0` (experimental but usable)

## License

MIT
