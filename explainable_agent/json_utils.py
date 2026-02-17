from __future__ import annotations

import json
import re
from typing import Any

try:
    from json_repair import repair_json
except Exception:  # noqa: BLE001
    repair_json = None


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
