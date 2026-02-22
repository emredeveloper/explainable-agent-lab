from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .schemas import RunTrace


def write_run_artifacts(trace: RunTrace, runs_dir: Path) -> tuple[Path, Path]:
    run_dir = runs_dir / trace.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / "trace.json"
    full_trace_path = run_dir / "trace_full.json"
    report_path = run_dir / "report.md"

    trace_path.write_text(
        json.dumps(_to_compact_trace(trace), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    full_trace_path.write_text(
        json.dumps(trace.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(_to_markdown_report(trace), encoding="utf-8")
    return trace_path, report_path


def _to_compact_trace(trace: RunTrace) -> dict[str, object]:
    duration_ms = _duration_ms(trace.started_at_utc, trace.finished_at_utc)
    steps: list[dict[str, object]] = []
    for step in trace.steps:
        steps.append(
            {
                "step": step.step,
                "action": step.decision.action,
                "source": step.audit.get("source"),
                "why_tool": _compact(step.audit.get("why_tool", ""), 180),
                "warnings": list(step.audit.get("warnings", [])),
                "tool": step.decision.tool_name,
                "tool_input": _compact(step.decision.tool_input or "", 120)
                if step.decision.tool_name
                else None,
                "error_analysis": step.decision.error_analysis,
                "proposed_fix": step.decision.proposed_fix,
                "tool_output_preview": _compact(step.tool_output or "", 180)
                if step.tool_output is not None
                else None,
                "confidence": round(step.decision.confidence, 3),
                "rationale": _compact(step.decision.rationale, 180),
                "latency_ms": step.latency_ms,
                "model_output_length": step.model_output_length,
                "tool_output_length": step.tool_output_length,
            }
        )
    return {
        "trace_version": "compact-v2",
        "run_id": trace.run_id,
        "task": trace.task,
        "model": {
            "requested": trace.requested_model,
            "resolved": trace.resolved_model,
        },
        "timing": {
            "started_at_utc": trace.started_at_utc,
            "finished_at_utc": trace.finished_at_utc,
            "duration_ms": duration_ms,
        },
        "final_answer": trace.final_answer,
        "faithfulness": {
            "likely_faithful": trace.faithfulness.likely_faithful,
            "lexical_similarity": round(trace.faithfulness.lexical_similarity, 3),
            "tool_support_score": round(trace.faithfulness.tool_support_score, 3),
            "note": trace.faithfulness.note,
        },
        "steps": steps,
        "errors": list(trace.errors),
        "efficiency_diagnostics": list(trace.efficiency_diagnostics),
    }


def _to_markdown_report(trace: RunTrace) -> str:
    lines: list[str] = []
    lines.append("# Explainable Agent Run Report")
    lines.append("")
    lines.append(f"- Run ID: `{trace.run_id}`")
    lines.append(f"- Requested model: `{trace.requested_model}`")
    lines.append(f"- Resolved model: `{trace.resolved_model}`")
    lines.append(f"- Started at: `{trace.started_at_utc}`")
    lines.append(f"- Finished at: `{trace.finished_at_utc}`")
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(trace.task)
    lines.append("")
    lines.append("## Final Answer")
    lines.append("")
    lines.append(trace.final_answer or "(empty)")
    lines.append("")
    lines.append("## Faithfulness Check")
    lines.append("")
    lines.append(f"- Likely faithful: `{trace.faithfulness.likely_faithful}`")
    lines.append(
        f"- Lexical similarity: `{trace.faithfulness.lexical_similarity:.3f}` "
        f"(threshold `{trace.faithfulness.threshold:.2f}`)"
    )
    lines.append(
        f"- Tool support score: `{trace.faithfulness.tool_support_score:.3f}` "
        f"(threshold `{trace.faithfulness.support_threshold:.2f}`)"
    )
    lines.append(f"- Note: {trace.faithfulness.note}")
    lines.append(f"- Alternative answer: {trace.faithfulness.alternative_answer}")
    lines.append("")
    lines.append("## Step Log")
    lines.append("")
    for step in trace.steps:
        lines.append(f"### Step {step.step}")
        lines.append(f"- Action: `{step.decision.action}`")
        if step.audit.get("source"):
            lines.append(f"- Decision source: `{step.audit.get('source')}`")
        if step.audit.get("why_tool"):
            lines.append(f"- Why this tool/action: {step.audit.get('why_tool')}")
        lines.append(f"- Rationale: {step.decision.rationale}")
        lines.append(f"- Confidence: `{step.decision.confidence:.2f}`")
        lines.append(f"- Evidence: {', '.join(step.decision.evidence)}")
        if step.decision.error_analysis:
            lines.append(f"- **Error Analysis (Self-Correction):** {step.decision.error_analysis}")
        if step.decision.proposed_fix:
            lines.append(f"- **Proposed Fix:** {step.decision.proposed_fix}")
        if step.audit.get("notes"):
            lines.append(f"- Notes: {', '.join(step.audit.get('notes', []))}")
        if step.audit.get("warnings"):
            lines.append(f"- Warnings: {', '.join(step.audit.get('warnings', []))}")
        if step.decision.tool_name:
            lines.append(f"- Tool: `{step.decision.tool_name}`")
            lines.append(f"- Tool input: `{step.decision.tool_input}`")
        if step.tool_output is not None:
            lines.append(f"- Tool output: `{_compact(step.tool_output)}`")
        lines.append(f"- Latency: `{step.latency_ms} ms`")
        lines.append(f"- Model output length: `{step.model_output_length} chars`")
        lines.append(f"- Tool output length: `{step.tool_output_length} chars`")
        lines.append("")
    if trace.errors:
        lines.append("## Errors")
        lines.append("")
        for err in trace.errors:
            lines.append(f"- {err}")
        lines.append("")

    lines.append("## Agent Diagnostics and Improvement Suggestions")
    lines.append("")
    diagnostics = _generate_diagnostics(trace)
    if not diagnostics:
        lines.append("- Agent completed the task without issues, no specific improvement suggestion.")
    else:
        for diag in diagnostics:
            lines.append(f"- {diag}")
    lines.append("")

    return "\n".join(lines)


def _generate_diagnostics(trace: RunTrace) -> list[str]:
    suggestions: list[str] = []
    
    # 1. Self-Correction Success Analysis
    error_steps = [s for s in trace.steps if s.decision.error_analysis]
    if error_steps:
        suggestions.append(f"Agent encountered an error {len(error_steps)} times and used self-correction ability.")
        last_error = error_steps[-1]
        suggestions.append(f"  > Last encountered issue: '{last_error.decision.error_analysis}'")
        suggestions.append(f"  > Proposed fix: '{last_error.decision.proposed_fix}'")
    
    # 2. Repeated tool usage (Looping/Stuck)
    tool_names = [s.decision.tool_name for s in trace.steps if s.decision.tool_name]
    if len(tool_names) > 2:
        for i in range(len(tool_names) - 2):
            if tool_names[i] == tool_names[i+1] == tool_names[i+2]:
                suggestions.append(f"WARNING: Agent called the `{tool_names[i]}` tool 3 times in a row. This might be a sign of an infinite loop. Consider adding clearer instructions to the prompt regarding this tool.")
                break

    # 3. Low Confidence Analysis
    low_conf_steps = [s for s in trace.steps if s.decision.confidence < 0.5]
    if low_conf_steps:
        avg_conf = sum(s.decision.confidence for s in low_conf_steps) / len(low_conf_steps)
        suggestions.append(f"Agent showed very low confidence score ({avg_conf:.2f}) in some steps. Consider providing more context to the system or extra information tools like web search.")

    # 4. Faithfulness Analysis
    if trace.faithfulness.tool_support_score > 0 and not trace.faithfulness.likely_faithful:
        suggestions.append("FAITHFULNESS WARNING: Agent used the tool successfully but the final answer does not sufficiently overlap with the tool output (Hallucination risk). Add the rule 'Only use the data coming from the tool, do not add your own interpretation' to the prompt.")
        
    # 5. Efficiency Diagnostics
    if hasattr(trace, 'efficiency_diagnostics') and trace.efficiency_diagnostics:
        suggestions.extend(trace.efficiency_diagnostics)

    return suggestions


def _compact(text: str, max_len: int = 240) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]..."


def _duration_ms(started_at_utc: str, finished_at_utc: str) -> int:
    try:
        started = datetime.fromisoformat(started_at_utc)
        finished = datetime.fromisoformat(finished_at_utc)
    except ValueError:
        return 0
    delta = finished - started
    return max(int(delta.total_seconds() * 1000), 0)
