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
                "warnings": list(step.audit.get("warnings", [])),
                "tool": step.decision.tool_name,
                "tool_input": _compact(step.decision.tool_input or "", 120)
                if step.decision.tool_name
                else None,
                "tool_output_preview": _compact(step.tool_output or "", 180)
                if step.tool_output is not None
                else None,
                "confidence": round(step.decision.confidence, 3),
                "rationale": _compact(step.decision.rationale, 180),
                "latency_ms": step.latency_ms,
            }
        )
    return {
        "trace_version": "compact-v1",
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
    }


def _to_markdown_report(trace: RunTrace) -> str:
    lines: list[str] = []
    lines.append("# Aciklanabilir Ajan Calisma Raporu")
    lines.append("")
    lines.append(f"- Calisma ID: `{trace.run_id}`")
    lines.append(f"- Istenen model: `{trace.requested_model}`")
    lines.append(f"- Kullanilan model: `{trace.resolved_model}`")
    lines.append(f"- Baslangic: `{trace.started_at_utc}`")
    lines.append(f"- Bitis: `{trace.finished_at_utc}`")
    lines.append("")
    lines.append("## Gorev")
    lines.append("")
    lines.append(trace.task)
    lines.append("")
    lines.append("## Nihai Cevap")
    lines.append("")
    lines.append(trace.final_answer or "(bos)")
    lines.append("")
    lines.append("## Faithfulness Kontrolu")
    lines.append("")
    lines.append(f"- Muhtemel faithfulness: `{trace.faithfulness.likely_faithful}`")
    lines.append(
        f"- Sozcuk benzerligi: `{trace.faithfulness.lexical_similarity:.3f}` "
        f"(esik `{trace.faithfulness.threshold:.2f}`)"
    )
    lines.append(
        f"- Arac destek skoru: `{trace.faithfulness.tool_support_score:.3f}` "
        f"(esik `{trace.faithfulness.support_threshold:.2f}`)"
    )
    lines.append(f"- Not: {trace.faithfulness.note}")
    lines.append(f"- Alternatif cevap: {trace.faithfulness.alternative_answer}")
    lines.append("")
    lines.append("## Adim Logu")
    lines.append("")
    for step in trace.steps:
        lines.append(f"### Step {step.step}")
        lines.append(f"- Eylem: `{step.decision.action}`")
        if step.audit.get("source"):
            lines.append(f"- Karar kaynagi: `{step.audit.get('source')}`")
        lines.append(f"- Gerekce: {step.decision.rationale}")
        lines.append(f"- Guven: `{step.decision.confidence:.2f}`")
        lines.append(f"- Kanit: {', '.join(step.decision.evidence)}")
        if step.audit.get("notes"):
            lines.append(f"- Notlar: {', '.join(step.audit.get('notes', []))}")
        if step.audit.get("warnings"):
            lines.append(f"- Uyarilar: {', '.join(step.audit.get('warnings', []))}")
        if step.decision.tool_name:
            lines.append(f"- Arac: `{step.decision.tool_name}`")
            lines.append(f"- Arac girdisi: `{step.decision.tool_input}`")
        if step.tool_output is not None:
            lines.append(f"- Arac cikisi: `{_compact(step.tool_output)}`")
        lines.append(f"- Gecikme: `{step.latency_ms} ms`")
        lines.append("")
    if trace.errors:
        lines.append("## Hatalar")
        lines.append("")
        for err in trace.errors:
            lines.append(f"- {err}")
        lines.append("")
    return "\n".join(lines)


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
