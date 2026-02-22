from __future__ import annotations

import re
import random
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .config import Settings
from .openai_client import OpenAICompatClient
from .schemas import Decision, FaithfulnessCheck, RunTrace, StepTrace
from .tools import (
    available_tool_names,
    run_tool,
    tool_catalog_payload,
    tool_catalog_text,
    tools_without_input,
)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}\"'")
    }


def lexical_jaccard_similarity(text_a: str, text_b: str) -> float:
    a = _tokenize(text_a)
    b = _tokenize(text_b)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


STOPWORDS = {
    "ve",
    "ile",
    "bir",
    "bu",
    "icin",
    "olarak",
    "the",
    "and",
    "or",
    "to",
    "of",
    "is",
    "are",
    "a",
    "an",
    "in",
    "on",
    "for",
}


def _content_tokens(text: str) -> set[str]:
    tokens = _tokenize(text)
    return {
        tok
        for tok in tokens
        if len(tok) >= 3 and tok not in STOPWORDS and not tok.isdigit()
    }


def tool_support_score(answer: str, steps: list[StepTrace]) -> float:
    tool_text = "\n".join(step.tool_output or "" for step in steps)
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    tool_tokens = _content_tokens(tool_text)
    if not tool_tokens:
        return 0.0
    overlap = answer_tokens & tool_tokens
    return len(overlap) / len(answer_tokens)


def _heuristic_tool_suggestion(task: str) -> tuple[str, str] | None:
    explicit = _extract_explicit_tool_request(task)
    if explicit:
        return explicit

    sql = _extract_sql_statement(task)
    if sql:
        if _is_read_only_sql(sql):
            return ("sqlite_query", sql)
        return ("sqlite_execute", sql)

    expression = _extract_math_expression(task)
    if expression:
        return ("calculate_math", expression)

    glob = _extract_glob_pattern(task)
    if glob:
        return ("list_workspace_files", f".|{glob}")

    path_candidate = _extract_path_candidate(task)
    if path_candidate:
        return ("read_text_file", path_candidate)

    return None


def _extract_explicit_tool_request(task: str) -> tuple[str, str] | None:
    lower = task.lower()
    known_tools = available_tool_names()
    without_input = tools_without_input()
    for tool in sorted(known_tools, key=len, reverse=True):
        idx = lower.find(tool)
        if idx == -1:
            continue
        raw_after = task[idx + len(tool) :]
        raw_input = raw_after.lstrip(" :|-").strip()

        if tool in without_input:
            return (tool, "")
        if tool == "list_workspace_files":
            return (tool, raw_input or ".")
        if tool == "calculate_math":
            if raw_input:
                return (tool, raw_input)
            expression = _extract_math_expression(task)
            if expression:
                return (tool, expression)
            return None
        if tool in {"sqlite_query", "sqlite_execute"}:
            if raw_input:
                return (tool, raw_input)
            sql = _extract_sql_statement(task)
            if sql:
                return (tool, sql)
            return None
        if raw_input:
            return (tool, raw_input)
    return None


def _extract_math_expression(text: str) -> str | None:
    candidates = re.findall(r"[0-9\.\s\+\-\*\/\(\)\^\%]{3,}", text)
    if not candidates:
        return None
    candidate = max(candidates, key=len).strip()
    if not any(op in candidate for op in "+-*/^%"):
        return None
    cleaned = re.sub(r"\s+", "", candidate)
    return cleaned.replace("^", "**")


def _extract_sql_statement(text: str) -> str | None:
    match = re.search(
        r"(?is)\b(select|with|pragma|insert|update|delete|create|drop|alter)\b.*",
        text,
    )
    if not match:
        return None
    statement = match.group(0).strip()
    statement = statement.strip("`\"'")
    return statement


def _is_read_only_sql(sql: str) -> bool:
    first = sql.strip().split()
    token = first[0].lower() if first else ""
    return token in {"select", "with", "pragma", "explain"}


def _extract_glob_pattern(text: str) -> str | None:
    match = re.search(r"\*[.][A-Za-z0-9_]+", text)
    if not match:
        return None
    return match.group(0)


def _extract_path_candidate(text: str) -> str | None:
    # Basit dosya yolu tespiti: or. docs/a.txt, .\data\file.csv
    match = re.search(
        r"([A-Za-z0-9_\-./\\]+[.][A-Za-z0-9]{1,8})",
        text,
    )
    if not match:
        return None
    return match.group(1)


def _should_override_first_tool(
    current_tool_name: str | None,
    suggested_tool_name: str,
) -> bool:
    current = (current_tool_name or "").strip()
    if not current:
        return True
    return current != suggested_tool_name


def _is_low_quality_answer(answer: str) -> bool:
    text = answer.strip().lower().strip(".!?,;:")
    if not text:
        return True
    if text.startswith("{") and ("\"action\"" in text or "'action'" in text):
        return True
    if text in {"final_answer", "tool_call"}:
        return True
    if text in available_tool_names():
        return True
    return len(text) <= 2


def _looks_generic_completion(answer: str) -> bool:
    raw = answer.strip()
    if not raw:
        return True
    normalized = raw.lower().strip(".!?,;:")
    words = normalized.split()
    if len(words) > 3:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    if any(sym in raw for sym in {"|", "/", "\\", ":", "\n", "{", "}", "[", "]"}):
        return False
    return True


def _fallback_answer_from_tool_outputs(steps: list[StepTrace]) -> str | None:
    tool_steps = [step for step in steps if step.decision.action == "tool_call"]
    if not tool_steps:
        return None

    latest_step = tool_steps[-1]
    tool_name = latest_step.decision.tool_name or ""
    raw_output = latest_step.tool_output or ""
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]

    if tool_name == "calculate_math":
        return lines[0] if lines else "Calculation result empty."

    if tool_name == "list_workspace_files":
        top = [ln for ln in lines if ln != "(empty)"][:5]
        if not top:
            return "Listing result empty."
        return "Found files:\n" + "\n".join(f"- {item}" for item in top)

    if tool_name == "read_text_file":
        preview = "\n".join(lines[:12]) if lines else "(empty content)"
        return "File preview:\n" + preview

    if tool_name.startswith("sqlite_"):
        top = lines[:12]
        if not top:
            return "SQLite result empty."
        return "SQLite result:\n" + "\n".join(top)

    return raw_output[:400] if raw_output else None


def _build_step_audit(
    decision: Decision,
    source: str,
    notes: list[str] | None = None,
    tool_output: str | None = None,
    why_tool: str | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if decision.confidence < 0.4:
        warnings.append("Low confidence detected.")
    if decision.action == "tool_call" and not decision.tool_name:
        warnings.append("tool_name is missing for tool_call action.")
    if decision.action == "final_answer":
        if _looks_generic_completion(decision.answer or ""):
            warnings.append("Final answer looks generic.")
        if _is_low_quality_answer(decision.answer or ""):
            warnings.append("Final answer might be low quality.")
    if tool_output and tool_output.startswith("ERROR:"):
        warnings.append("Tool execution returned error.")
    return {
        "source": source,
        "why_tool": why_tool or "",
        "notes": list(notes or []),
        "warnings": warnings,
    }


def _analyze_efficiency(steps: list[StepTrace]) -> list[str]:
    diagnostics: list[str] = []
    
    if len(steps) >= 5:
        diagnostics.append(f"Analysis: It took the agent {len(steps)} steps to reach a conclusion. Complex multi-step paths increase the risk of hallucination. Consider breaking down the task or creating higher-level tools.")
        
    for step in steps:
        if step.tool_output_length > 4000:
            diagnostics.append(f"Warning: The output from '{step.decision.tool_name}' at Step {step.step} was extremely long ({step.tool_output_length} characters). This may overwhelm the model's context window and degrade performance. Consider adding a tool that filters or summarizes this data.")
            
    return diagnostics


class ExplainableAgent:
    def __init__(self, settings: Settings, client: OpenAICompatClient | None = None, verbose: bool = False) -> None:
        self.settings = settings
        self.client = client or OpenAICompatClient(
            base_url=settings.base_url, api_key=settings.api_key
        )
        self.verbose = verbose

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        if self.settings.chaos_mode and random.random() < 0.2:
            chaos_errors = [
                "ERROR: [Chaos Mode] TIMEOUT: The external service did not respond in time.",
                "ERROR: [Chaos Mode] MALFORMED_DATA: Received unreadable garbage instead of JSON.",
                "ERROR: [Chaos Mode] PERMISSION_DENIED: Agent lacks the required role to run this tool.",
                "ERROR: [Chaos Mode] RATE_LIMIT: Too many requests, please retry later or use another tool."
            ]
            return random.choice(chaos_errors)

        return run_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            workspace_root=self.settings.workspace_root,
        )

    def _print_step(self, step_num: int, decision: Decision, tool_output: str | None = None):
        if not self.verbose:
            return
            
        action_color = "bold blue" if decision.action == "tool_call" else "bold green"
        
        content = Text()
        content.append(f"Rationale: ", style="bold")
        content.append(f"{decision.rationale}\n")
        
        conf_color = "red" if decision.confidence < 0.5 else "yellow" if decision.confidence < 0.8 else "green"
        content.append(f"Confidence: ", style="bold")
        content.append(f"{decision.confidence:.2f}\n", style=conf_color)
        
        if decision.action == "tool_call":
            content.append(f"Tool: ", style="bold")
            content.append(f"{decision.tool_name}\n", style="magenta")
            content.append(f"Input: ", style="bold")
            content.append(f"{decision.tool_input}\n")
            if tool_output:
                out_preview = tool_output[:300] + "..." if len(tool_output) > 300 else tool_output
                content.append(f"Output: ", style="bold")
                content.append(f"{out_preview}\n", style="dim")
        else:
            content.append(f"Final Answer: ", style="bold")
            content.append(f"{decision.answer}\n", style="bold white")
            
        if decision.error_analysis:
            content.append(f"\nError Analysis: ", style="bold red")
            content.append(f"{decision.error_analysis}\n")
        if decision.proposed_fix:
            content.append(f"Proposed Fix: ", style="bold yellow")
            content.append(f"{decision.proposed_fix}\n")

        panel = Panel(
            content,
            title=f"Step {step_num} | Action: {decision.action}",
            border_style=action_color,
            expand=False
        )
        console.print(panel)

    def run(self, task: str) -> RunTrace:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[
            :8
        ]
        started_at = _utc_now()
        resolved_model = self.client.resolve_model(self.settings.requested_model)

        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"User task:\n{task}\n\n"
                    f"Available tools:\n{tool_catalog_text()}\n\n"
                    f"Tool catalog JSON:\n{tool_catalog_payload()}\n\n"
                    "Select only one tool per step if necessary."
                ),
            }
        ]

        steps: list[StepTrace] = []
        errors: list[str] = []
        final_answer = ""
        explicit_tool_request = _extract_explicit_tool_request(task)

        loop_start = 1
        if explicit_tool_request:
            tool_name, tool_input = explicit_tool_request
            tool_output = self._execute_tool(
                tool_name=tool_name,
                tool_input=tool_input,
            )
            steps.append(
                StepTrace(
                    step=1,
                    model_output="[deterministic_tool_call]",
                    decision=Decision(
                        action="tool_call",
                        rationale="Explicit tool request applied directly.",
                        confidence=1.0,
                        evidence=[
                            "Tool name explicitly provided in task text.",
                            "First step routed to deterministic tool.",
                        ],
                        tool_name=tool_name,
                        tool_input=tool_input,
                    ),
                    tool_output=tool_output,
                    latency_ms=0,
                    model_output_length=len("[deterministic_tool_call]"),
                    tool_output_length=len(tool_output or ""),
                    audit={
                        "source": "explicit_request",
                        "why_tool": "Tool name explicitly mentioned in user text.",
                        "notes": [
                            "LLM step skipped because explicit tool name found in task."
                        ],
                        "warnings": [],
                    },
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Deterministic tool call executed: {tool_name}\n"
                        f"Input: {tool_input}\nOutput:\n{tool_output}"
                    ),
                }
            )

            explicit_next_prompt = "Provide a short final answer based on the tool result. Return decision in valid JSON format."
            if tool_output and tool_output.startswith("ERROR:"):
                explicit_next_prompt = "ERROR received from previous tool. Please state the cause of the error and how to fix it using the 'error_analysis' and 'proposed_fix' fields in the JSON. WARNING: You MUST call a new tool by setting action='tool_call' to fix the error. Also, DO NOT FORGET to fill in the 'tool_name' and 'tool_input' fields in the JSON!"

            messages.append(
                {
                    "role": "user",
                    "content": explicit_next_prompt,
                }
            )
            self._print_step(1, steps[0].decision, tool_output)
            loop_start = 2

        for step in range(loop_start, self.settings.max_steps + 1):
            decision, raw_output, latency_ms = self.client.get_decision(
                model=resolved_model,
                messages=messages,
                temperature=self.settings.temperature,
                reasoning_effort=self.settings.reasoning_effort,
            )
            decision_source = "model"
            decision_notes: list[str] = []

            suggestion = _heuristic_tool_suggestion(task) if not steps else None
            if suggestion and decision.action == "tool_call":
                tool_name, tool_input = suggestion
                if _should_override_first_tool(decision.tool_name, tool_name):
                    decision = Decision(
                        action="tool_call",
                        rationale=(
                            "Heuristic correction: tool selected according to task signal in first step."
                        ),
                        confidence=max(decision.confidence, 0.7),
                        evidence=[
                            "Distinct structural signal detected in first step.",
                            "Deterministic tool selection applied for more stable execution.",
                        ],
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                    raw_output = raw_output + "\n[heuristic_override_tool_correction]"
                    decision_source = "heuristic_override"
                    decision_notes.append("Heuristic correction for tool applied in first step.")

            if suggestion and decision.action == "final_answer":
                tool_name, tool_input = suggestion
                decision = Decision(
                    action="tool_call",
                    rationale=(
                        "Heuristic correction: task has direct math/SQL/file signal."
                    ),
                    confidence=max(decision.confidence, 0.7),
                    evidence=[
                        "Structural task signal requires tool usage.",
                        "Early final answer prevented in first step.",
                    ],
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
                raw_output = raw_output + "\n[heuristic_override_tool_call]"
                decision_source = "heuristic_override"
                decision_notes.append(
                    "Early final prevented in first step, forced tool call."
                )

            if decision.action == "tool_call":
                tool_output = self._execute_tool(
                    tool_name=decision.tool_name or "",
                    tool_input=decision.tool_input or "",
                )
                steps.append(
                    StepTrace(
                        step=step,
                        model_output=raw_output,
                        decision=decision,
                        tool_output=tool_output,
                        latency_ms=latency_ms,
                        model_output_length=len(raw_output or ""),
                        tool_output_length=len(tool_output or ""),
                        audit=_build_step_audit(
                            decision=decision,
                            source=decision_source,
                            notes=decision_notes,
                            tool_output=tool_output,
                            why_tool=(
                                f"{decision.tool_name} selected; rationale: {decision.rationale}"
                                if decision.tool_name
                                else ""
                            ),
                        ),
                    )
                )
                self._print_step(step, decision, tool_output)
                messages.append({"role": "assistant", "content": raw_output})
                content_text = f"Tool result ('{decision.tool_name}'):\n{tool_output}\n\n"
                if tool_output and tool_output.startswith("ERROR:"):
                    content_text += "ERROR received from previous tool. Please use the 'error_analysis' and 'proposed_fix' fields in the JSON to state the cause of the error and how to fix it. WARNING: You MUST call a new tool by setting action='tool_call' to fix the error. Only use final_answer if you absolutely cannot find a solution. Also, DO NOT FORGET to fill in the 'tool_name' and 'tool_input' fields in the JSON!"
                else:
                    content_text += "Select the next step."

                messages.append(
                    {
                        "role": "user",
                        "content": content_text,
                    }
                )
                continue

            final_answer = decision.answer or ""
            steps.append(
                StepTrace(
                    step=step,
                    model_output=raw_output,
                    decision=decision,
                    tool_output=None,
                    latency_ms=latency_ms,
                    model_output_length=len(raw_output or ""),
                    tool_output_length=0,
                    audit=_build_step_audit(
                        decision=decision,
                        source=decision_source,
                        notes=decision_notes,
                        tool_output=None,
                        why_tool="Final answer selected, new tool call not required.",
                    ),
                )
            )
            self._print_step(step, decision)
            break

        if not final_answer:
            fallback = _fallback_answer_from_tool_outputs(steps)
            if fallback:
                final_answer = fallback
                errors.append(
                    "Model did not provide a final answer within max steps; generated from tool result."
                )
            else:
                errors.append(
                    "Maximum steps reached; final answer regenerated from model."
                )
                final_answer = self.client.get_alternative_answer(
                    model=resolved_model,
                    task=task,
                    temperature=self.settings.temperature,
                    reasoning_effort=self.settings.reasoning_effort,
                )

        tool_used = any(step.decision.action == "tool_call" for step in steps)
        
        if tool_used and _looks_generic_completion(final_answer):
            fallback = _fallback_answer_from_tool_outputs(steps)
            if fallback:
                final_answer = fallback
                errors.append(
                    "Generic/invalid final answer automatically corrected from tool result."
                )
        if _is_low_quality_answer(final_answer):
            if tool_used:
                fallback = _fallback_answer_from_tool_outputs(steps)
                if fallback:
                    final_answer = fallback
                    errors.append("Low quality final answer automatically corrected from tool result.")
            else:
                final_answer = self.client.get_alternative_answer(
                    model=resolved_model,
                    task=task,
                    temperature=self.settings.temperature,
                    reasoning_effort=self.settings.reasoning_effort,
                )
                errors.append("Low quality final answer refreshed with direct answer.")

        if tool_used:
            alternative_answer = self.client.get_alternative_answer(
                model=resolved_model,
                task=task,
                temperature=self.settings.temperature,
                reasoning_effort=self.settings.reasoning_effort,
            )
            similarity = lexical_jaccard_similarity(final_answer, alternative_answer)
            threshold = 0.75
            support_threshold = 0.25
            support_score = tool_support_score(final_answer, steps)
            likely_faithful = similarity < threshold or support_score >= support_threshold
            note = (
                "Strong tool trace (low alternative similarity or high tool overlap)."
                if likely_faithful
                else "Weak tool trace: high alternative similarity and low tool overlap."
            )
        else:
            alternative_answer = "(skipped check: tool not used)"
            similarity = 1.0
            threshold = 0.75
            support_threshold = 0.25
            support_score = 0.0
            likely_faithful = False
            note = "Faithfulness check skipped because there is no tool call."

        finished_at = _utc_now()
        faithfulness = FaithfulnessCheck(
            alternative_answer=alternative_answer,
            lexical_similarity=similarity,
            threshold=threshold,
            likely_faithful=likely_faithful,
            note=note,
            tool_support_score=support_score,
            support_threshold=support_threshold,
        )
        
        diagnostics = _analyze_efficiency(steps)

        return RunTrace(
            run_id=run_id,
            task=task,
            requested_model=self.settings.requested_model,
            resolved_model=resolved_model,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            steps=steps,
            final_answer=final_answer,
            faithfulness=faithfulness,
            errors=errors,
            efficiency_diagnostics=diagnostics,
        )
