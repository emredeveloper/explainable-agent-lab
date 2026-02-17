from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from .schemas import Decision


DECISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_decision",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["tool_call", "final_answer"]},
                "rationale": {"type": "string"},
                "confidence": {"type": "number"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "tool_name": {"type": ["string", "null"]},
                "tool_input": {"type": ["string", "null"]},
                "answer": {"type": ["string", "null"]},
            },
            "required": ["action", "rationale", "confidence", "evidence"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

class LMStudioClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def list_models(self) -> list[str]:
        models = self.client.models.list()
        return [m.id for m in models.data]

    def resolve_model(self, requested_model: str) -> str:
        model_ids = self.list_models()
        if not model_ids:
            raise RuntimeError(
                "LM Studio API ulasilabilir ama yuklu model yok. "
                "Modeli LM Studio'da yukleyip tekrar deneyin."
            )
        if requested_model in model_ids:
            return requested_model
        requested_lower = requested_model.lower()
        partial_matches = [mid for mid in model_ids if requested_lower in mid.lower()]
        if partial_matches:
            return partial_matches[0]
        available = ", ".join(model_ids)
        raise RuntimeError(
            f"Istenen model bulunamadi: '{requested_model}'. "
            f"Yuklu modeller: {available}"
        )

    def get_decision(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        reasoning_effort: str,
    ) -> tuple[Decision, str, int]:
        system_prompt = self._build_decision_prompt(reasoning_effort)
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                response_format=DECISION_SCHEMA,
            )
        except Exception:  # noqa: BLE001
            response = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content or ""
        payload = self._parse_json_payload(content)
        decision = self._to_decision(payload, fallback_text=content)
        return decision, content, latency_ms

    def get_alternative_answer(
        self,
        model: str,
        task: str,
        temperature: float,
        reasoning_effort: str,
    ) -> str:
        system_prompt = self._build_final_prompt(reasoning_effort)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _build_decision_prompt(reasoning_effort: str) -> str:
        return f"""Aciklanabilir bir ajan planlayicisisin.
Reasoning effort: {reasoning_effort}.
Her turda sadece tek bir sonraki adimi sec ve sadece JSON dondur.

Kurallar:
- Dis bilgi, dosya, SQL ya da hesaplama gerekiyorsa action="tool_call" sec.
- Bilgi yeterliyse action="final_answer" sec.
- confidence 0 ile 1 arasinda olmali.
- evidence alanina kararini destekleyen somut sinyalleri yaz.
- action="tool_call" ise tool_name ve tool_input doldur.
- action="final_answer" ise answer doldur.
- Cevapta dusunce zinciri veya ekstra metin verme; yalniz JSON ver.
"""

    @staticmethod
    def _build_final_prompt(reasoning_effort: str) -> str:
        return f"""Reasoning effort: {reasoning_effort}.
Kullanici gorevini arac kullanmadan dogrudan cevapla.
Yanit kisa, net ve Turkce olsun."""

    @staticmethod
    def _parse_json_payload(content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            return {}
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        candidate = LMStudioClient._extract_first_json_object(content)
        if candidate:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        # Fallback parser for models that return prose around JSON.
        stack = 0
        start = -1
        for idx, char in enumerate(text):
            if char == "{":
                if stack == 0:
                    start = idx
                stack += 1
            elif char == "}":
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start != -1:
                        return text[start : idx + 1]
        # Last fallback: regex for simple single object.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else None

    @staticmethod
    def _to_decision(payload: dict[str, Any], fallback_text: str) -> Decision:
        action = str(payload.get("action", "final_answer")).strip().lower()
        if action not in {"tool_call", "final_answer"}:
            action = "final_answer"

        rationale = str(payload.get("rationale", "")).strip() or "Gerekce belirtilmedi."
        raw_confidence = payload.get("confidence", 0.5)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        raw_evidence = payload.get("evidence", [])
        if isinstance(raw_evidence, list):
            evidence = [str(item) for item in raw_evidence if str(item).strip()]
        else:
            evidence = [str(raw_evidence)] if str(raw_evidence).strip() else []
        if not evidence:
            evidence = ["Acik kanit belirtilmedi."]

        tool_name = payload.get("tool_name")
        tool_input = payload.get("tool_input")
        answer = payload.get("answer")

        if action == "tool_call" and not tool_name:
            action = "final_answer"
            answer = (
                str(answer).strip()
                if answer
                else "Arac cagrisi istendi ama tool_name eksik."
            )
        if action == "final_answer":
            if not answer:
                answer = fallback_text.strip() or "Nihai cevap saglanmadi."
            return Decision(
                action="final_answer",
                rationale=rationale,
                confidence=confidence,
                evidence=evidence,
                answer=str(answer).strip(),
            )

        return Decision(
            action="tool_call",
            rationale=rationale,
            confidence=confidence,
            evidence=evidence,
            tool_name=str(tool_name).strip(),
            tool_input=str(tool_input or "").strip(),
        )
