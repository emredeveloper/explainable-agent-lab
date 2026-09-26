from __future__ import annotations

import ast
import json
import re
from typing import Any

try:
    from json_repair import repair_json
except Exception:  # noqa: BLE001
    repair_json = None


def _tool_call_payload(
    tool_name: str, tool_input: str, protocol: str
) -> dict[str, Any]:
    return {
        "action": "tool_call",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "rationale": f"Tool call decoded from the model's {protocol} protocol.",
        "confidence": 0.5,
        "evidence": ["A single tool call was decoded from the model response."],
    }


def _decode_tool_input(raw: str) -> str:
    value = raw.strip()
    if value in {"", "{}", "null"}:
        return ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError, RecursionError):
            return value
    if isinstance(parsed, (str, int, float, bool)):
        return str(parsed)
    return value


def parse_text_decision(text: str) -> dict[str, Any] | None:
    """Normalize common structured text responses without evaluating model code."""
    match = re.fullmatch(
        r"\s*<\|tool_call_start\|>(.*?)<\|tool_call_end\|>\s*", text, re.DOTALL
    )
    if match:
        try:
            node = ast.parse(match.group(1).strip(), mode="eval").body
            if isinstance(node, ast.List) and len(node.elts) == 1:
                node = node.elts[0]
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                return None
            if len(node.args) + len(node.keywords) > 1:
                return None
            if any(kw.arg is None for kw in node.keywords):
                return None
            arg = (
                node.args[0]
                if node.args
                else node.keywords[0].value
                if node.keywords
                else None
            )
            value = ast.literal_eval(arg) if arg is not None else ""
            if not isinstance(value, (str, int, float, bool)):
                return None
        except (SyntaxError, ValueError, TypeError, RecursionError):
            return None
        return _tool_call_payload(node.func.id, str(value), "tagged text")

    xml_starts = list(re.finditer(r"<tool_call\b[^>]*>", text, re.I))
    if xml_starts:
        if len(xml_starts) != 1:
            return None
        start = xml_starts[0].end()
        end = re.search(r"</tool_call>", text[start:], re.I)
        body = text[start : start + end.start()] if end else text[start:]
        fields = re.findall(
            r"<(?:function|parameter)=(action|tool_name|tool_input|answer)>"
            r"(.*?)</(?:function|parameter)>",
            body,
            re.DOTALL | re.I,
        )
        values: dict[str, list[str]] = {}
        for key, value in fields:
            values.setdefault(key.casefold(), []).append(value.strip())
        if len(values.get("action", [])) > 1 or len(values.get("answer", [])) > 1:
            return None
        action = values.get("action", ["tool_call"])[0].casefold()
        if action == "final_answer":
            answers = values.get("answer", [])
            if not answers or not answers[0].strip():
                return None
            return {
                "action": "final_answer",
                "answer": answers[0].strip(),
                "rationale": "Final answer decoded from XML-like model fields.",
                "confidence": 0.5,
                "evidence": ["The model supplied an explicit final-answer field."],
            }
        if action != "tool_call":
            return None
        if (
            len(values.get("tool_name", [])) != 1
            or len(values.get("tool_input", [])) != 1
        ):
            return None
        name = values["tool_name"][0]
        if not re.fullmatch(r"[A-Za-z_][\w.-]{0,63}", name):
            return None
        raw_input = values.get("tool_input", [""])[0]
        return _tool_call_payload(name, _decode_tool_input(raw_input), "XML-like text")

    final_answer = parse_text_final_answer(text)
    if final_answer is not None:
        return final_answer

    action_lines = re.findall(
        r"(?im)^\s*action\s*[:=]\s*[\"']?([A-Za-z_]+)[\"']?\s*$", text
    )
    name_lines = re.findall(
        r"(?im)^\s*tool_name\s*[:=]\s*[\"']?([^\"'\r\n]+?)[\"']?\s*$",
        text,
    )
    input_lines = re.findall(r"(?im)^\s*tool_input\s*[:=]\s*(.*?)\s*$", text)
    if (
        len(action_lines) > 1
        or (action_lines and action_lines[0].casefold() != "tool_call")
        or len(name_lines) != 1
        or len(input_lines) != 1
    ):
        return None
    name = name_lines[0].strip()
    if not re.fullmatch(r"[A-Za-z_][\w.-]{0,63}", name):
        return None
    raw_input = input_lines[0] if input_lines else ""
    return _tool_call_payload(name, _decode_tool_input(raw_input), "field-based text")


def parse_text_final_answer(text: str) -> dict[str, Any] | None:
    """Extract an explicit final-answer field from common non-JSON responses."""
    actions = re.findall(
        r"(?im)^\s*action\s*[:=]\s*[\"']?([A-Za-z_]+)[\"']?\s*$", text
    )
    if actions and (len(actions) != 1 or actions[0].casefold() != "final_answer"):
        return None
    if not actions and re.search(r"(?im)^\s*tool_(?:name|input)\s*[:=]", text):
        return None

    lines = text.splitlines()
    answer_rows = [
        (index, re.match(r"^\s*answer\s*[:=]\s*(.*)$", line, re.I))
        for index, line in enumerate(lines)
    ]
    answer_rows = [(index, match) for index, match in answer_rows if match]
    if len(answer_rows) != 1:
        return None

    index, answer_match = answer_rows[0]
    answer_parts = [answer_match.group(1)] if answer_match.group(1) else []
    for line in lines[index + 1 :]:
        if re.match(
            r"^\s*(?:action|tool_name|tool_input|rationale|confidence|evidence|error_analysis|proposed_fix)\s*[:=]",
            line,
            re.I,
        ):
            break
        answer_parts.append(line)
    answer = "\n".join(answer_parts).strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {"'", '"'}:
        try:
            decoded = json.loads(answer) if answer[0] == '"' else ast.literal_eval(answer)
            if isinstance(decoded, str):
                answer = decoded
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            pass
    if not answer:
        return None
    return {
        "action": "final_answer",
        "answer": answer,
        "rationale": "Final answer decoded from model text fields.",
        "confidence": 0.5,
        "evidence": ["The model supplied an explicit final-answer field."],
    }


def parse_text_tool_call(text: str) -> dict[str, Any] | None:
    """Compatibility helper that returns only normalized tool-call decisions."""
    decision = parse_text_decision(text)
    return decision if decision and decision.get("action") == "tool_call" else None


def extract_first_json_object(text: str) -> str | None:
    stack = 0
    start = -1
    for idx, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = idx
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start != -1:
                    return text[start : idx + 1]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def parse_json_object_relaxed(text: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = (text or "").strip()
    if not raw:
        return None, "empty"
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload, None
    except json.JSONDecodeError:
        pass

    candidate = extract_first_json_object(raw)
    if candidate:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload, "extracted"
        except json.JSONDecodeError:
            pass

    if repair_json is not None:
        try:
            repaired = repair_json(raw, return_objects=True)
            if isinstance(repaired, dict):
                return repaired, "json_repair"
        except Exception:  # noqa: BLE001
            pass
        if candidate:
            try:
                repaired = repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict):
                    return repaired, "json_repair"
            except Exception:  # noqa: BLE001
                pass

    return None, "parse_error"
