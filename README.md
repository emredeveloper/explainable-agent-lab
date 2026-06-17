# 🔬 Explainable Agent Lab

> A local-first, explainable agent framework designed to guide developers in building robust AI agents.

Building reliable agents is hard. LLMs hallucinate, get stuck in infinite loops, or fail to parse tools correctly. **Explainable Agent Lab** is built to solve this by focusing on **explainability and guidance**.

✨ **Key Features:**
- **Show the Hidden Errors:** Reveal exactly where and why an agent fails (e.g., low confidence, schema violations).
- **Self-Healing:** The agent automatically analyzes its own errors and proposes alternative tool-based solutions.
- **Visual Terminal Tracking:** Step-by-step interactive and colorful tracking using the `rich` library (`--verbose`).
- **Detailed Diagnostic Reports:** Actionable suggestions on hallucination risks, loop patterns, and prompt improvements.
- **Chaos Engineering (Stress Testing):** Inject simulated tool errors (e.g., timeouts, missing data) to test your agent's self-healing capabilities.
- **Efficiency Diagnostics:** Track token usage and step counts to identify context window exhaustion and prompt inefficiencies.
- **Multi-Agent Orchestration (Team of Thoughts):** Coordinate multiple specialized agents with transparent delegation plans, sub-agent traces, and orchestration diagnostics.

---

## 🚀 Quick Start

### 1. Install
Install directly from PyPI:
```bash
pip install explainable-agent
```

PyPI currently publishes `0.3.0` (released June 17, 2026). If PyPI is behind a future GitHub release, install the repository version instead:
```bash
pip install "git+https://github.com/emredeveloper/explainable-agent-lab.git@v0.3.0"
```

For development, clone the repo and run:
```bash
pip install -e ".[dev]"
```

Check the installed package version with:
```bash
python -c "import explainable_agent; print(explainable_agent.__version__)"
```

### 2. Connect Your Local LLM
You can use any OpenAI-compatible local server like **Ollama** or **LM Studio**.

- **Ollama:** `http://localhost:11434/v1` (e.g., model: `qwen3.5:9b`)
- **LM Studio:** `http://localhost:1234/v1` (e.g., model: `google/gemma-3-12b`)

*Tip: You can create a `.env` file in your working directory to set your defaults (see `.env.example`).*

### 3. Run the Agent
The package installs a global CLI command `explainable-agent`.

**Example using Ollama:**
```bash
explainable-agent \
  --base-url http://localhost:11434/v1 \
  --model qwen3.5:9b \
  --task "calculate_math: (215*4)-12" \
  --verbose
```

---

## 💻 Using the Python API

Easily integrate the agent into your codebase and inspect structured traces.

```python
from explainable_agent import ExplainableAgent, Settings, write_run_artifacts

settings = Settings.from_env().with_overrides(
    base_url="http://localhost:1234/v1",
    api_key="local",
    requested_model="google/gemma-4-e4b",
)

agent = ExplainableAgent(settings=settings, verbose=False)
trace = agent.run("calculate_math: (215*4)-12")

print(trace.final_answer)
write_run_artifacts(trace, settings.runs_dir)
```

### Custom tools

For simple applications, register a global tool with `@define_tool`:

```python
from pathlib import Path

from explainable_agent import define_tool


@define_tool(
    name="echo_upper",
    description="Converts text to uppercase.",
    usage_hint="Input is plain text.",
)
def echo_upper(text: str, workspace_root: Path) -> str:
    return text.upper()
```

For libraries, tests, and multi-tenant applications, prefer an isolated `ToolRegistry`
so tool registrations do not leak across agent instances:

```python
from pathlib import Path

from explainable_agent import ExplainableAgent, Settings, ToolRegistry

registry = ToolRegistry.from_global()


@registry.define_tool(
    name="echo_upper",
    description="Converts text to uppercase.",
    usage_hint="Input is plain text.",
)
def echo_upper(text: str, workspace_root: Path) -> str:
    return text.upper()


agent = ExplainableAgent(
    settings=Settings.from_env(),
    tool_registry=registry,
)
```

The stable public API exports `ExplainableAgent`, `Settings`, `ToolRegistry`,
`ToolSpec`, `define_tool`, `run_tool`, trace dataclasses, and artifact writers.

Check out the `examples/` directory:
- [`examples/basic_usage.py`](examples/basic_usage.py) - Small default smoke run using `.env`/CLI settings.
- [`examples/custom_tool_usage.py`](examples/custom_tool_usage.py) - Register a custom Python tool and call it through the agent.
- [`examples/showcase_all_features.py`](examples/showcase_all_features.py) - Compact local showcase for math, SQLite, custom tools, optional chaos mode, and optional JSONL eval.
- [`examples/multi_agent_demo.py`](examples/multi_agent_demo.py) - Multi-agent orchestration with a researcher and SQLite specialist.

Run quick examples with Ollama:
```bash
python examples/basic_usage.py \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --model qwen3.5:9b

python examples/showcase_all_features.py \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --model qwen3.5:9b
```

Use `--include-sqlite`, `--include-custom`, `--include-chaos`, or `--include-eval` on the showcase when you want the slower optional scenarios.

---

## 📊 Evaluation & Custom Datasets

Evaluate your fine-tuned models or custom datasets easily. The pipeline parses messy outputs, repairs broken JSON, and generates actionable Markdown reports.

- **Custom JSONL datasets:**  
  1. Create a `.jsonl` dataset (see `examples/custom_eval_sample.jsonl`).  
  2. Run the evaluation:
     ```bash
     python scripts/eval_hf_tool_calls.py \
       --dataset examples/custom_eval_sample.jsonl \
       --model qwen3.5:9b
     ```

- **Built-in HF-style tool-calling sample (JSONL):**
  A small complex function-calling benchmark is bundled under `data/evals/hf_complexfuncbench_first_turn_100.jsonl`.
  Example with LM Studio and `google/gemma-3-12b`:
  ```bash
  python scripts/eval_hf_tool_calls.py \
    --dataset data/evals/hf_complexfuncbench_first_turn_100.jsonl \
    --base-url http://localhost:1234/v1 \
    --model google/gemma-3-12b \
    --limit 10 \
    --sampling head
  ```

We also support standard benchmarks out of the box:
- **HF Tool Calls:** `data/evals/hf_complexfuncbench_first_turn_100.jsonl`
- **BFCL SQL:** `data/evals/bfcl_sql/BFCL_v3_sql.json`
- **SWE-bench Lite:** `data/evals/swebench_lite_test.jsonl`

---

## 🔍 Tracing & Verbosity Modes

The agent supports two primary verbosity modes:

- **Verbose mode (`verbose=True` or `--verbose`):**
  - Prints an **Agent tools flow roadmap** at the start (task, model, config, available tools, and control flow).
  - Shows rich, per-step panels including:
    - Decision source (`model`, `explicit_request`, `heuristic_override`)
    - Latency per step
    - Rationale, confidence, tool name/input/output
    - Error analysis and proposed fix (for self-healing steps)
  - Ends with a **developer run summary** panel (tool flow recap, faithfulness note, efficiency diagnostics).

- **Concise mode (`verbose=False`):**
  - Prints a one-line **flow summary** (e.g., `Step 1: calculate_math [FAIL] -> Step 2: calculate_math [OK] -> Step 3: final_answer`).
  - Shows total step count, self-healed error count, a short final answer preview, and key warnings (if any).

---

## 🛠️ Built-in Tools
The agent comes with out-of-the-box tools ready to use:
`duckduckgo_search`, `calculate_math`, `read_text_file`, `list_workspace_files`, `now_utc`, `sqlite_init_demo`, `sqlite_list_tables`, `sqlite_describe_table`, `sqlite_query`, `sqlite_execute`.

`duckduckgo_search` remains the tool name in the API, while the underlying search dependency is provided by `ddgs`.

---
*License: MIT | Current Release: v0.3.0*

<!-- profile-priority: current-ai-agent-project -->
