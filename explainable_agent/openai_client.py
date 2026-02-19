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
                "error_analysis": {"type": ["string", "null"]},
                "proposed_fix": {"type": ["string", "null"]},
            },
            "required": [
                "action",
                "rationale",
                "confidence",
                "evidence",
                "tool_name",
                "tool_input",
                "answer",
                "error_analysis",
                "proposed_fix"
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def list_models(self) -> list[str]:
        models = self.client.models.list()
        return [m.id for m in models.data]

    def resolve_model(self, requested_model: str) -> str:
        model_ids = self.list_models()
        if not model_ids:
            raise RuntimeError(
                "API reachable but no model loaded. Please load a model on the server and try again."
            )
        if requested_model in model_ids:
            return requested_model
        requested_lower = requested_model.lower()
        partial_matches = [mid for mid in model_ids if requested_lower in mid.lower()]
        if partial_matches:
            return partial_matches[0]
        available = ", ".join(model_ids)
        raise RuntimeError(
            f"Requested model not found: '{requested_model}'. Loaded models: {available}"
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
        return f"""You are an explainable agent planner.
Reasoning effort: {reasoning_effort}.
Select only one next step per turn and return ONLY JSON.

Rules:
- If external info, file, SQL, or calculation is needed, set action="tool_call".
- If you have enough info, set action="final_answer".
- confidence must be between 0 and 1.
- Write concrete signals supporting your decision in the evidence field.
- If action="tool_call", fill in tool_name and tool_input.
- If action="final_answer", fill in answer.
- If you got an ERROR from the previous tool:
  1. Briefly write the reason for the error in "error_analysis".
  2. Briefly write how to fix it in "proposed_fix".
  3. Then you MUST set "action": "tool_call" to fix the ERROR and write the new tool's name in "tool_name". The error won't be fixed unless you call a new tool!
  4. Also, make sure to fill the "tool_input" field for the new tool.
  5. Only if absolutely no tool can fix the error, set "action": "final_answer" and explain the situation in "answer".
- Do not include any chain of thought or extra text in your response; output ONLY JSON.
"""

    @staticmethod
    def _build_final_prompt(reasoning_effort: str) -> str:
        return f"""Reasoning effort: {reasoning_effort}.
Answer the user task directly without using tools.
The response should be short, clear, and in English."""

    @staticmethod
    def _parse_json_payload(content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            return {}

        # Remove markdown code block fences if present (e.g. ```json ... ```)
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 2:
                # Remove the first line (```json) and the last line (```)
                content = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        candidate = OpenAICompatClient._extract_first_json_object(content)
        if candidate:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                # Fallback to relax parsing for LLMs if standard fails
                pass

        try:
            from json_repair import repair_json
            repaired = repair_json(content, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            
            if candidate:
                repaired_candidate = repair_json(candidate, return_objects=True)
                if isinstance(repaired_candidate, dict):
                    return repaired_candidate
        except ImportError:
            pass
        except Exception:
            pass

        return {}

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
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
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else None

    @staticmethod
    def _to_decision(payload: dict[str, Any], fallback_text: str) -> Decision:
        action = str(payload.get("action", "final_answer")).strip().lower()
        if action not in {"tool_call", "final_answer"}:
            action = "final_answer"

        rationale = str(payload.get("rationale", "")).strip() or "No rationale provided."
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
            evidence = ["No explicit evidence provided."]

        tool_name = payload.get("tool_name")
        tool_input = payload.get("tool_input")
        answer = payload.get("answer")
        error_analysis = payload.get("error_analysis")
        proposed_fix = payload.get("proposed_fix")

        # Eğer error_analysis ve proposed_fix dolduysa ve action tool_call ama name eksikse,
        # aracı doğrudan sqlite_describe_table olarak düzeltmeye çalışmak yerine modelden geleni kullansın,
        # fakat name eksikse bir sorun var. Biz yine de action tool_call ama tool_name eksik durumunu final_answer'a çeviriyoruz.
        # Aslında hatayı gördüyse ve çözüm sunduysa yeni bir tool call yapabilmeliydi.
        # Eger hata analizini yazdiysa ve "action" = "tool_call" verdiyse ama "tool_name" unuttuysa
        # veya "action" = "final_answer" verdi ama "tool_name" de verdiyse (ikincisi garip),
        # biz her durumda asagidaki kontrolu yapiyoruz:
        if error_analysis and proposed_fix and action == "final_answer":
            # Model belki bir arac cagirmak istedi ama yanlislikla action'i final_answer birakti
            # ya da "yeni arac belirtilmedi" mesajini yazdi.
            # Eger tool_name verdiyse onu dinleyip tool_call'a cevirebiliriz:
            if tool_name:
                action = "tool_call"
            else:
                # Eger tool_name vermediyse model gercekten durmustur, bir sey yapamayiz.
                pass

        if action == "tool_call" and not tool_name:
            action = "final_answer"
            answer = (
                str(answer).strip()
                if answer
                else (
                     "Error analysis done but no new tool specified." 
                     if error_analysis else "Tool call requested but tool_name is missing."
                )
            )
        if action == "final_answer":
            if not answer:
                answer = fallback_text.strip() or "No final answer provided."
            return Decision(
                action="final_answer",
                rationale=rationale,
                confidence=confidence,
                evidence=evidence,
                answer=str(answer).strip(),
                error_analysis=str(error_analysis).strip() if error_analysis else None,
                proposed_fix=str(proposed_fix).strip() if proposed_fix else None,
            )

        return Decision(
            action="tool_call",
            rationale=rationale,
            confidence=confidence,
            evidence=evidence,
            tool_name=str(tool_name).strip(),
            tool_input=str(tool_input or "").strip(),
            error_analysis=str(error_analysis).strip() if error_analysis else None,
            proposed_fix=str(proposed_fix).strip() if proposed_fix else None,
        )
