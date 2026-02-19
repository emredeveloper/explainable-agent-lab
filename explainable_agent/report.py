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
        if step.audit.get("why_tool"):
            lines.append(f"- Neden bu arac/aksiyon: {step.audit.get('why_tool')}")
        lines.append(f"- Gerekce: {step.decision.rationale}")
        lines.append(f"- Guven: `{step.decision.confidence:.2f}`")
        lines.append(f"- Kanit: {', '.join(step.decision.evidence)}")
        if step.decision.error_analysis:
            lines.append(f"- **Hata Analizi (Self-Correction):** {step.decision.error_analysis}")
        if step.decision.proposed_fix:
            lines.append(f"- **Onerilen Cozum:** {step.decision.proposed_fix}")
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

    lines.append("## Ajan Teshis ve Iyilestirme Onerileri (Diagnostics)")
    lines.append("")
    diagnostics = _generate_diagnostics(trace)
    if not diagnostics:
        lines.append("- Ajan gorevi sorunsuz tamamladi, ozel bir iyilestirme onerisi yok.")
    else:
        for diag in diagnostics:
            lines.append(f"- {diag}")
    lines.append("")

    return "\n".join(lines)


def _generate_diagnostics(trace: RunTrace) -> list[str]:
    suggestions: list[str] = []
    
    # 1. Self-Correction (Hata Ayiklama) Basarisi Analizi
    error_steps = [s for s in trace.steps if s.decision.error_analysis]
    if error_steps:
        suggestions.append(f"Ajan {len(error_steps)} kez hata ile karsilasti ve self-correction (kendi kendini duzeltme) yetenegini kullandi.")
        last_error = error_steps[-1]
        suggestions.append(f"  > Son karsilasilan sorun: '{last_error.decision.error_analysis}'")
        suggestions.append(f"  > Onerilen duzeltme: '{last_error.decision.proposed_fix}'")
    
    # 2. Arka arkaya ayni aracin kullanilmasi (Looping/Stuck)
    tool_names = [s.decision.tool_name for s in trace.steps if s.decision.tool_name]
    if len(tool_names) > 2:
        for i in range(len(tool_names) - 2):
            if tool_names[i] == tool_names[i+1] == tool_names[i+2]:
                suggestions.append(f"DIKKAT: Ajan `{tool_names[i]}` aracini arka arkaya 3 kez cagirdi. Bu bir sonsuz dongu (loop) isareti olabilir. Prompt'a bu aracla ilgili daha net talimatlar ekleyin.")
                break

    # 3. Dusuk Guven (Confidence) Analizi
    low_conf_steps = [s for s in trace.steps if s.decision.confidence < 0.5]
    if low_conf_steps:
        avg_conf = sum(s.decision.confidence for s in low_conf_steps) / len(low_conf_steps)
        suggestions.append(f"Ajan bazi adimlarda cok dusuk guven skoru ({avg_conf:.2f}) sergiledi. Sisteme daha fazla baglam (context) veya web aramasi gibi ekstra bilgi araclari saglamayi dusunun.")

    # 4. Faithfulness (Sadakat) Analizi
    if trace.faithfulness.tool_support_score > 0 and not trace.faithfulness.likely_faithful:
        suggestions.append("SADAKAT UYARISI: Ajan araci basariyla kullandi fakat verdigi nihai cevap arac ciktisiyla yeterince ortusmuyor (Halusinasyon riski). Prompt'a 'Sadece aractan gelen veriyi kullan, kendi yorumunu katma' kuralini ekleyin.")
        
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
