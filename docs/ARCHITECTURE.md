# Architecture

This document describes the high-level architecture of **Explainable Agent Lab**.

---

## Core modules

- `explainable_agent/agent.py`
  - Main orchestration loop for a single agent.
  - Decision → tool call → next decision control flow.
  - Trace/audit generation, faithfulness checks, efficiency diagnostics.
  - Verbosity handling for rich vs. concise terminal output.

- `explainable_agent/openai_client.py`
  - OpenAI-compatible client wrapper.
  - Builds structured decision prompts and parses model JSON into `Decision` objects.
  - Provides alternative-answer queries for faithfulness checks and low-quality answer recovery.

- `explainable_agent/tools.py`
  - Built-in tools and guards (file / SQLite / math / web).
  - Tool registry (`define_tool`, `AVAILABLE_TOOLS`) and runtime execution helper.

- `explainable_agent/report.py`
  - Run artifact writers:
    - `trace.json` (compact trace)
    - `trace_full.json` (full, verbose trace)
    - `report.md` (human-readable diagnostic report)
  - Orchestrator report writer for multi-agent runs (`orch_trace.json`, `orch_report.md`).

- `explainable_agent/schemas.py`
  - `Decision`, `StepTrace`, `RunTrace`, `FaithfulnessCheck` dataclasses.
  - Orchestrator-specific schemas: `SubTaskTrace`, `OrchestratorRunTrace`.

- `explainable_agent/config.py`
  - `Settings` dataclass and `.from_env()` helper.
  - Central place for base URL, model, workspace, runs directory, SQLite path, chaos mode, etc.

- `explainable_agent/orchestrator.py`
  - `TeamOrchestrator` for multi-agent “team of thoughts” workflows.
  - Generates delegation plans, runs sub-agents, and synthesizes final answers.

- `explainable_agent/eval_tool_calls.py`
  - Tool-call normalization and scoring logic for HF-style tool-calling benchmarks.
  - Handles JSONL datasets, relaxed JSON parsing, and argument-level scoring.

- `explainable_agent/dataset_adapters.py`
  - Dataset adapters for SWE-bench Lite and other formats.
  - Normalizes heterogeneous JSON/JSONL datasets into a common internal schema.

- `explainable_agent/json_utils.py`
  - Robust JSON parsing utilities (`parse_json_object_relaxed`) with repair strategies.

---

## Single-agent runtime flow

1. **Task ingestion**
   - Task arrives via CLI (`explainable-agent`) or Python API (`ExplainableAgent.run`).
   - The agent builds an initial user message that includes:
     - The user task
     - A textual tool catalog
     - A JSON tool catalog payload

2. **Decision step**
   - The agent calls `OpenAICompatClient.get_decision(...)` with:
     - System prompt describing the decision schema.
     - Conversation history (including previous tool outputs).
   - The model returns a JSON decision payload:
     - `action`: `"tool_call"` or `"final_answer"`.
     - `tool_name`, `tool_input` (for tool calls).
     - `rationale`, `confidence`, `evidence`.
     - Optional `error_analysis` and `proposed_fix` fields for self-healing.

3. **Heuristic override (first step only)**
   - A heuristic layer examines the original task for:
     - Explicit tool names (e.g., `calculate_math:`, `sqlite_init_demo:`).
     - SQL, math expressions, glob patterns, or file paths.
   - If the first model decision conflicts with a strong signal, the agent may:
     - Override the tool choice or force a tool call instead of an early final answer.

4. **Tool execution**
   - If `action == "tool_call"`:
     - The agent executes the selected tool via `run_tool` with workspace guards.
     - Tool output is stored in a `StepTrace`.
     - If chaos mode is enabled, synthetic errors may be injected for robustness testing.

5. **Self-healing loop**
   - If a tool output starts with `"ERROR:"`, the agent:
     - Encourages the model (via prompts) to:
       - Explain the error in `error_analysis`.
       - Suggest a fix in `proposed_fix`.
       - Call a new tool with corrected inputs.
   - This yields visible self-healing behavior in the step traces.

6. **Final answer and quality checks**
   - When `action == "final_answer"`:
     - A `StepTrace` is recorded without a tool output.
     - If no final answer is produced within `max_steps`, a fallback path is used
       (regenerating from tools or via a direct alternative answer call).
   - Faithfulness check:
     - Compares the final answer with an alternative answer from the model.
     - Measures lexical similarity and a tool-support score to estimate whether
       the answer is grounded in tool outputs.
   - Efficiency diagnostics:
     - Flags long tool outputs and multi-step paths that may harm performance.

7. **Artifacts and terminal output**
   - A `RunTrace` object is returned with full trace information.
   - `report.write_run_artifacts` writes JSON/Markdown artifacts into `runs/<run_id>/`.
   - Verbosity modes:
     - `verbose=True`: rich step panels, roadmap, and developer summary.
     - `verbose=False`: concise single-line flow + short answer and warnings.

---

## Multi-agent orchestration flow

1. **Team setup**
   - Multiple `ExplainableAgent` instances are created with shared or customized settings.
   - A `TeamOrchestrator` is constructed with a mapping of agent IDs to:
     - Natural language capability descriptions.
     - The corresponding agent instance.

2. **Delegation planning**
   - Given a high-level task, the orchestrator:
     - Uses the model to propose a delegation plan (subtasks per agent).
     - Produces a human-readable plan describing which agent does what.

3. **Sub-agent execution**
   - For each subtask, the orchestrator:
     - Routes the task to the appropriate agent.
     - Captures the resulting `RunTrace` as a `SubTaskTrace`.

4. **Final synthesis**
   - The orchestrator asks the model to synthesize a final answer from:
     - The original task.
     - All sub-agent traces and intermediate results.
   - The orchestration trace and Markdown report are written via
     `write_orchestrator_artifacts` in `report.py`.

---

## Evaluation flows (JSONL and benchmarks)

### HF-style tool-calling evaluation (`scripts/eval_hf_tool_calls.py`)

1. Load dataset rows from JSONL (e.g., `data/evals/hf_complexfuncbench_first_turn_100.jsonl`).
2. Prompt the model to return `tool_calls` in structured JSON.
3. Parse, repair, and normalize predicted tool calls.
4. Score the predictions against expected tool calls (name, count, arguments).
5. Write `summary.json`, `details.json`, and `report.md` under `runs/evals/`.

### BFCL SQL evaluation

1. Load BFCL SQL task and answer files from JSON.
2. Normalize SQL tool calls for DELETE/INSERT/UPDATE/SELECT variants.
3. Score predictions with lenient rules for column ordering and optional fields.

### SWE-bench Lite readiness (`scripts/eval_swebench_readiness.py`)

1. Use `dataset_adapters` to load `swebench_lite` JSONL rows.
2. Prompt the model to emit a JSON plan with:
   - `analysis.root_cause_hypothesis`
   - `files_to_inspect`
   - `first_actions`
3. Parse and validate the plan with relaxed JSON parsing.
4. Compute:
   - Parse success rate.
   - Percentage of plans with complete analysis + files + actions.
5. Write readiness metrics and details to `runs/evals/`.
