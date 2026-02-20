# 🔬 Explainable Agent Lab

> A local-first, explainable agent framework designed to guide developers in building robust AI agents.

Building reliable agents is hard. LLMs hallucinate, get stuck in infinite loops, or fail to parse tools correctly. **Explainable Agent Lab** is built to solve this by focusing on **explainability and guidance**.

✨ **Key Features:**
- **Show the Hidden Errors:** Reveal exactly where and why an agent fails (e.g., low confidence, schema violations).
- **Self-Healing:** The agent automatically analyzes its own errors and proposes alternative tool-based solutions.
- **Visual Terminal Tracking:** Step-by-step interactive and colorful tracking using the `rich` library (`--verbose`).
- **Detailed Diagnostic Reports:** Actionable suggestions on hallucination risks, loop patterns, and prompt improvements.

---

## 🚀 Quick Start

### 1. Install
Install directly from PyPI:
```bash
pip install explainable-agent
```

*(Optional: for development, clone the repo and run `pip install -e .[dev]`)*

### 2. Connect Your Local LLM
You can use any OpenAI-compatible local server like **Ollama** or **LM Studio**.

- **Ollama:** `http://localhost:11434/v1` (e.g., model: `ministral-3:14b`)
- **LM Studio:** `http://localhost:1234/v1` (e.g., model: `gpt-oss-20b`)

*Tip: You can create a `.env` file in your working directory to set your defaults (see `.env.example`).*

### 3. Run the Agent
The package installs a global CLI command `explainable-agent`.

**Example using Ollama:**
```bash
explainable-agent \
  --base-url http://localhost:11434/v1 \
  --model ministral-3:14b \
  --task "calculate_math: (215*4)-12" \
  --verbose
```

---

## 💻 Using the Python API

Easily integrate the agent into your codebase or create custom tools using the `@define_tool` decorator. 

Check out the `examples/` directory:
- [`examples/basic_usage.py`](examples/basic_usage.py) - Initialize and run the agent programmatically.
- [`examples/custom_tool_usage.py`](examples/custom_tool_usage.py) - Learn how to build custom tools and watch the agent self-heal from errors.

Run an example:
```bash
python examples/custom_tool_usage.py
```

---

## 📊 Evaluation & Custom Datasets

Evaluate your fine-tuned models or custom datasets easily. The pipeline parses messy outputs, repairs broken JSON, and generates actionable Markdown reports.

**1. Create a `.jsonl` dataset** (See `examples/custom_eval_sample.jsonl`)

**2. Run the evaluation:**
```bash
python scripts/eval_hf_tool_calls.py \
  --dataset examples/custom_eval_sample.jsonl \
  --model ministral-3:14b
```

We also support standard benchmarks out of the box:
- **HF Tool Calls:** `data/evals/hf_xlam_fc_sample.jsonl`
- **BFCL SQL:** `data/evals/bfcl_sql/BFCL_v3_sql.json`
- **SWE-bench Lite:** `data/evals/swebench_lite_test.jsonl`

---

## 🛠️ Built-in Tools
The agent comes with out-of-the-box tools ready to use:
`duckduckgo_search`, `calculate_math`, `read_text_file`, `list_workspace_files`, `now_utc`, `sqlite_init_demo`, `sqlite_list_tables`, `sqlite_describe_table`, `sqlite_query`, `sqlite_execute`.

---
*License: MIT | Current Release: v0.1.0*