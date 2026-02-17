from __future__ import annotations

import re
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
        return lines[0] if lines else "Hesaplama sonucu bos."

    if tool_name == "list_workspace_files":
        top = [ln for ln in lines if ln != "(bos)"][:5]
        if not top:
            return "Listeleme sonucu bos."
        return "Bulunan dosyalar:\n" + "\n".join(f"- {item}" for item in top)

    if tool_name == "read_text_file":
        preview = "\n".join(lines[:12]) if lines else "(bos icerik)"
        return "Dosya onizlemesi:\n" + preview

    if tool_name.startswith("sqlite_"):
        top = lines[:12]
        if not top:
            return "SQLite sonucu bos."
        return "SQLite sonucu:\n" + "\n".join(top)

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
        warnings.append("Dusuk confidence tespit edildi.")
    if decision.action == "tool_call" and not decision.tool_name:
        warnings.append("tool_call aksiyonu icin tool_name eksik.")
    if decision.action == "final_answer":
        if _looks_generic_completion(decision.answer or ""):
            warnings.append("Nihai cevap genel gorunuyor.")
        if _is_low_quality_answer(decision.answer or ""):
            warnings.append("Nihai cevap dusuk kaliteli olabilir.")
    if tool_output and tool_output.startswith("ERROR:"):
        warnings.append("Arac calismasi hata dondu.")
    return {
        "source": source,
        "why_tool": why_tool or "",
        "notes": list(notes or []),
        "warnings": warnings,
    }


class ExplainableAgent:
    def __init__(self, settings: Settings, client: OpenAICompatClient | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAICompatClient(
            base_url=settings.base_url, api_key=settings.api_key
        )

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
                    f"Kullanici gorevi:\n{task}\n\n"
                    f"Kullanilabilir araclar:\n{tool_catalog_text()}\n\n"
                    f"Arac katalog JSON'u:\n{tool_catalog_payload()}\n\n"
                    "Gerekirse her adimda yalnizca bir arac sec."
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
            tool_output = run_tool(
                tool_name=tool_name,
                tool_input=tool_input,
                workspace_root=self.settings.workspace_root,
            )
            steps.append(
                StepTrace(
                    step=1,
                    model_output="[deterministic_tool_call]",
                    decision=Decision(
                        action="tool_call",
                        rationale="Acik arac istegi dogrudan uygulandi.",
                        confidence=1.0,
                        evidence=[
                            "Gorev metninde arac adi acikca verildi.",
                            "Ilk adim deterministik araca yonlendirildi.",
                        ],
                        tool_name=tool_name,
                        tool_input=tool_input,
                    ),
                    tool_output=tool_output,
                    latency_ms=0,
                    audit={
                        "source": "explicit_request",
                        "why_tool": "Kullanicinin metninde arac adi acikca gecti.",
                        "notes": [
                            "Gorevde acik arac adi bulundugu icin LLM adimi atlandi."
                        ],
                        "warnings": [],
                    },
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Deterministik arac cagrisi yapildi: {tool_name}\n"
                        f"Girdi: {tool_input}\nCikti:\n{tool_output}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Arac sonucuna gore Turkce ve kisa bir nihai cevap ver. "
                        "JSON formatina uygun karar dondur."
                    ),
                }
            )
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
                            "Heuristik duzeltme: ilk adimda gorev sinyaline uygun arac secildi."
                        ),
                        confidence=max(decision.confidence, 0.7),
                        evidence=[
                            "Ilk adimda belirgin yapisal sinyal algilandi.",
                            "Daha stabil calisma icin deterministik arac secimi uygulandi.",
                        ],
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                    raw_output = raw_output + "\n[heuristic_override_tool_correction]"
                    decision_source = "heuristic_override"
                    decision_notes.append("Ilk adimda araca yonelik heuristik duzeltme uygulandi.")

            if suggestion and decision.action == "final_answer":
                tool_name, tool_input = suggestion
                decision = Decision(
                    action="tool_call",
                    rationale=(
                        "Heuristik duzeltme: gorevde dogrudan hesap/SQL/dosya sinyali var."
                    ),
                    confidence=max(decision.confidence, 0.7),
                    evidence=[
                        "Yapisal gorev sinyali arac kullanimini gerektiriyor.",
                        "Ilk adimda erken final yaniti engellendi.",
                    ],
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
                raw_output = raw_output + "\n[heuristic_override_tool_call]"
                decision_source = "heuristic_override"
                decision_notes.append(
                    "Ilk adimda erken final engellendi, arac cagrisi zorlandi."
                )

            if decision.action == "tool_call":
                tool_output = run_tool(
                    tool_name=decision.tool_name or "",
                    tool_input=decision.tool_input or "",
                    workspace_root=self.settings.workspace_root,
                )
                steps.append(
                    StepTrace(
                        step=step,
                        model_output=raw_output,
                        decision=decision,
                        tool_output=tool_output,
                        latency_ms=latency_ms,
                        audit=_build_step_audit(
                            decision=decision,
                            source=decision_source,
                            notes=decision_notes,
                            tool_output=tool_output,
                            why_tool=(
                                f"{decision.tool_name} secildi; gerekce: {decision.rationale}"
                                if decision.tool_name
                                else ""
                            ),
                        ),
                    )
                )
                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Arac sonucu ('{decision.tool_name}'):\n{tool_output}\n\n"
                            "Siradaki adimi sec."
                        ),
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
                    audit=_build_step_audit(
                        decision=decision,
                        source=decision_source,
                        notes=decision_notes,
                        tool_output=None,
                        why_tool="Final answer secildi, yeni tool cagrisi gerekli degil.",
                    ),
                )
            )
            break

        if not final_answer:
            fallback = _fallback_answer_from_tool_outputs(steps)
            if fallback:
                final_answer = fallback
                errors.append(
                    "Maksimum adimda model nihai cevap vermedi; arac sonucundan cevap uretildi."
                )
            else:
                errors.append(
                    "Maksimum adim sayisina ulasildi; nihai cevap modelden yeniden alindi."
                )
                final_answer = self.client.get_alternative_answer(
                    model=resolved_model,
                    task=task,
                    temperature=self.settings.temperature,
                    reasoning_effort=self.settings.reasoning_effort,
                )

        tool_used = any(step.decision.action == "tool_call" for step in steps)
        explicit_tool_requested = _extract_explicit_tool_request(task) is not None
        if tool_used and explicit_tool_requested:
            fallback = _fallback_answer_from_tool_outputs(steps)
            if fallback:
                final_answer = fallback
                errors.append(
                    "Acik arac gorevinde nihai cevap, arac sonucundan standartlastirildi."
                )
        if tool_used and _looks_generic_completion(final_answer):
            fallback = _fallback_answer_from_tool_outputs(steps)
            if fallback:
                final_answer = fallback
                errors.append(
                    "Genel/gecersiz nihai cevap, arac sonucundan otomatik duzeltildi."
                )
        if _is_low_quality_answer(final_answer):
            if tool_used:
                fallback = _fallback_answer_from_tool_outputs(steps)
                if fallback:
                    final_answer = fallback
                    errors.append("Dusuk kaliteli nihai cevap, arac sonucundan otomatik duzeltildi.")
            else:
                final_answer = self.client.get_alternative_answer(
                    model=resolved_model,
                    task=task,
                    temperature=self.settings.temperature,
                    reasoning_effort=self.settings.reasoning_effort,
                )
                errors.append("Dusuk kaliteli nihai cevap, dogrudan yanit ile yenilendi.")

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
                "Arac izi guclu (alternatif benzerlik dusuk veya arac-overlap yuksek)."
                if likely_faithful
                else "Arac izi zayif: alternatif benzerlik yuksek ve arac-overlap dusuk."
            )
        else:
            alternative_answer = "(atlanan kontrol: arac kullanilmadi)"
            similarity = 1.0
            threshold = 0.75
            support_threshold = 0.25
            support_score = 0.0
            likely_faithful = False
            note = "Faithfulness kontrolu atlandi cunku arac cagrisi yok."

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
        )
