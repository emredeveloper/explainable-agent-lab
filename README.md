# Explainable Agent Lab

Local-first, explainable agent framework designed to guide developers building AI agents.

The core mission of this project is **explainability and guidance**. Building reliable agents is hard; LLMs hallucinate, get stuck in infinite loops, or fail to parse tools correctly. This repository is built to:
1. **Show the hidden errors:** Reveal exactly where and why an agent fails (e.g., low confidence, schema violations, logic loops).
2. **Guide the builder:** Provide actionable diagnostics, error analysis, and improvement suggestions for your prompts and tools.
3. **Evaluate and iterate:** Run tool-using agents with transparent step-by-step traces and benchmark their performance on custom datasets.

## What You Get

- Structured agent decisions per step:
  - `action`, `confidence`, `rationale`, `evidence`, `error_analysis`, `proposed_fix`
- **Self-Healing:** The agent's ability to automatically analyze its own errors upon failure and propose alternative solutions.
- **Visual Terminal Tracking (Verbose Mode):** Step-by-step interactive and colorful tracking in the terminal using the `rich` library.
- **Detailed Diagnostic Reports:** Concrete suggestions at the end of each run regarding agent performance, error rates, hallucination, and infinite loop risks.
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

You can use any local LLM runner that supports OpenAI-compatible endpoints, such as **LM Studio** or **Ollama**.

### Connecting to LM Studio

1. Open LM Studio and start the local server.
2. Load a model (e.g. `gpt-oss-20b`).
3. The default endpoint is `http://localhost:1234/v1`.

### Connecting to Ollama

1. Install and start Ollama on your machine.
2. Run a model, for example: `ollama run ministral-3:14b`
3. The default endpoint is `http://localhost:11434/v1`.
4. You can pass this via CLI using `--base-url http://localhost:11434/v1` or update your `.env` file.

List available models on your active server:

```bash
python -m explainable_agent.cli --list-models
```

Run the agent (LM Studio example):

```bash
python -m explainable_agent.cli --model gpt-oss-20b --reasoning-effort high --task "calculate_math: (215*4)-12"
```

Run the agent (Ollama example):

```bash
python -m explainable_agent.cli --base-url http://localhost:11434/v1 --model ministral-3:14b --reasoning-effort high --task "calculate_math: (215*4)-12"
```

Run with visual trace in terminal:

```bash
python -m explainable_agent.cli --task "sqlite_query: SELECT name, email FROM customers" --verbose
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

You can easily evaluate your own custom datasets, HuggingFace datasets, or fine-tuned models. The evaluation pipeline automatically parses messy model outputs, attempts to repair broken JSON, and compares the predicted tool calls against your expected ones.

### Evaluating Custom Datasets

To evaluate a custom dataset or a dataset downloaded from HuggingFace, it simply needs to be in `.jsonl` (JSON Lines) format with the following structure per line:

```json
{
  "query": "What is the weather like in Tokyo right now?",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Fetches the current weather for a specified location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string" }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "expected_tool_calls": [
    {
      "name": "get_weather",
      "arguments": {
        "location": "Tokyo"
      }
    }
  ]
}
```

We have provided a sample in `examples/custom_eval_sample.jsonl`. You can run the evaluation against it using:

```bash
python scripts/eval_hf_tool_calls.py --dataset examples/custom_eval_sample.jsonl --model your-model-name
```

After the run, you will get a detailed Markdown report containing Error Breakdowns, Argument Match Rates, Failure Patterns, and an Actionable Plan.

## Using the Python API

You can easily integrate the Explainable Agent into your own Python codebase. We provide ready-to-run examples in the `examples/` directory.

### Basic Usage

You can initialize the agent, run tasks, and save diagnostic reports programmatically:

```bash
python examples/basic_usage.py
```

### Adding Custom Tools

The framework is highly extensible. You can register your own Python functions as agent tools using the `@define_tool` decorator. The agent will automatically discover them, understand their usage rules, and recover from any custom exceptions you raise.

See how to create a custom tool and run it:

```bash
python examples/custom_tool_usage.py
```

### Standard Benchmark Examples

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
