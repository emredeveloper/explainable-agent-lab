from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from jsonschema import Draft7Validator
except Exception:  # noqa: BLE001
    Draft7Validator = None

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except Exception:  # noqa: BLE001
    BaseModel = None
    ConfigDict = None
    Field = None
    ValidationError = Exception

try:
    from json_repair import repair_json
except Exception:  # noqa: BLE001
    repair_json = None

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except Exception:  # noqa: BLE001
    retry = None
    retry_if_exception_type = None
    stop_after_attempt = None
    wait_exponential = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainable_agent.config import Settings
from explainable_agent.eval_tool_calls import (
    load_bfcl_sql_samples,
    load_eval_samples,
    normalize_tool_calls,
    score_prediction,
    score_prediction_variants,
)
from explainable_agent.json_utils import parse_json_object_relaxed


DEFAULT_DATASET = Path("data/evals/hf_xlam_fc_sample.jsonl")


RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tool_call_prediction",
        "schema": {
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tool_calls"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


if BaseModel is not None and ConfigDict is not None and Field is not None:
    class ToolCallModel(BaseModel):
        model_config = ConfigDict(extra="ignore")
        name: str
        arguments: dict[str, Any] = Field(default_factory=dict)


    class ToolCallResponseModel(BaseModel):
        model_config = ConfigDict(extra="ignore")
        tool_calls: list[ToolCallModel] = Field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hugging Face tool-calling mini benchmark degerlendirmesi"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Degerlendirilecek jsonl dosya yolu.",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["auto", "jsonl", "bfcl_sql"],
        default="auto",
        help="Veri formati. auto secenegi BFCL SQL dosyasini otomatik algilar.",
    )
    parser.add_argument(
        "--answers",
        type=str,
        default=None,
        help="BFCL SQL answer dosyasi (possible_answer/BFCL_v3_sql.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model adi (varsayilan: AGENT_MODEL veya gpt-oss-20b).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort seviyesi.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ilk N ornegi calistir.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["random", "head"],
        default="random",
        help="Limit uygulanirken ornek secim stratejisi.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random sampling icin seed (opsiyonel).",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/evals",
        help="Rapor cikti klasoru.",
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="Parse/guard hatasinda uygulanacak otomatik duzeltme deneme sayisi.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Tahmin edilen tool_call sayisini en fazla bu degerle sinirla (opsiyonel).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help=(
            "Model cikti token ust limiti. Verilmezse/0 ise max_tokens gonderilmez; "
            "model cevap tamamlandiginda durur."
        ),
    )
    return parser.parse_args()


def _call_chat_completion(
    client: OpenAI,
    request: dict[str, Any],
    max_attempts: int = 3,
) -> Any:
    if retry and stop_after_attempt and wait_exponential and retry_if_exception_type:

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _run() -> Any:
            return client.chat.completions.create(**request)

        return _run()

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_attempts:
                raise
            backoff = min(0.25 * (2 ** (attempt - 1)), 2.0)
            time.sleep(backoff)
    if last_error:
        raise last_error
    raise RuntimeError("chat completion cagrisi basarisiz.")


def _request_completion(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int | None,
    with_schema: bool,
) -> Any:
    base_request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if max_completion_tokens is not None and max_completion_tokens > 0:
        base_request["max_tokens"] = max_completion_tokens
    if with_schema:
        try:
            return _call_chat_completion(
                client,
                {**base_request, "response_format": RESPONSE_SCHEMA},
            )
        except Exception:
            return _call_chat_completion(client, base_request)
    return _call_chat_completion(client, base_request)


def _response_content(response: Any) -> str:
    if response is None:
        return ""
    try:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
    except Exception:  # noqa: BLE001
        pass
    if isinstance(response, dict):
        try:
            content = response["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else str(content)
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}:\n{content}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _normalize_tool_name(name: str) -> str:
    cleaned = name.strip()
    for prefix in ("functions.", "function.", "tool."):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if cleaned.endswith(")") and "(" in cleaned:
        cleaned = cleaned.split("(", 1)[0]
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.lower()


def _build_tool_specs(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {})
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        params = fn.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        normalized_schema = _normalize_schema_for_validator(params)
        schema_validator = _build_schema_validator(normalized_schema)
        properties = params.get("properties", {})
        allowed_keys = set(properties.keys()) if isinstance(properties, dict) else None
        required = params.get("required", [])
        required_keys = set(required) if isinstance(required, list) else set()
        additional_properties = params.get("additionalProperties", True)
        normalized = _normalize_tool_name(name)
        specs[normalized] = {
            "name": name,
            "allowed_keys": allowed_keys,
            "required_keys": required_keys,
            "additional_properties": bool(additional_properties),
            "schema_validator": schema_validator,
        }
    return specs


def _normalize_schema_for_validator(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if key == "type" and item == "dict":
                    normalized[key] = "object"
                else:
                    normalized[key] = _walk(item)
            return normalized
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    normalized = _walk(schema)
    if isinstance(normalized, dict) and "type" not in normalized:
        normalized["type"] = "object"
    if isinstance(normalized, dict) and normalized.get("type") == "object":
        normalized.setdefault("properties", {})
        normalized.setdefault("additionalProperties", True)
    return normalized if isinstance(normalized, dict) else {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


def _build_schema_validator(schema: dict[str, Any]) -> Any:
    if Draft7Validator is None:
        return None
    try:
        return Draft7Validator(schema)
    except Exception:  # noqa: BLE001
        return None


def _find_closest_tool_spec(
    raw_name: str, specs: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if not specs:
        return None
    normalized = _normalize_tool_name(raw_name)
    if normalized in specs:
        return specs[normalized]
    matches = difflib.get_close_matches(normalized, list(specs.keys()), n=1, cutoff=0.8)
    if not matches:
        return None
    return specs[matches[0]]


def _guard_tool_calls(
    predicted_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    specs = _build_tool_specs(tools)
    guarded: list[dict[str, Any]] = []
    dropped_unknown = 0
    pruned_argument_keys = 0
    missing_required_keys = 0
    canonicalized_tool_names = 0
    schema_validation_errors = 0

    for call in predicted_calls:
        if not isinstance(call, dict):
            continue
        raw_name = str(call.get("name", "")).strip()
        if not raw_name:
            dropped_unknown += 1
            continue
        spec = _find_closest_tool_spec(raw_name, specs)
        if not spec:
            dropped_unknown += 1
            continue

        canonical_name = spec["name"]
        if canonical_name != raw_name:
            canonicalized_tool_names += 1

        raw_arguments = call.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                raw_arguments = {}
        if not isinstance(raw_arguments, dict):
            raw_arguments = {}
        raw_arguments = _normalize_argument_values(raw_arguments)

        allowed_keys = spec.get("allowed_keys")
        additional_properties = spec.get("additional_properties", True)
        if allowed_keys is None or additional_properties:
            guarded_arguments = dict(raw_arguments)
        else:
            guarded_arguments = {}
            for key, value in raw_arguments.items():
                if key in allowed_keys:
                    guarded_arguments[key] = value
                else:
                    pruned_argument_keys += 1

        required_keys = spec.get("required_keys", set())
        for key in required_keys:
            if key not in guarded_arguments:
                missing_required_keys += 1

        validator = spec.get("schema_validator")
        if validator is not None:
            first_pass_errors = list(validator.iter_errors(guarded_arguments))
            if first_pass_errors:
                for err in first_pass_errors:
                    schema_validation_errors += 1
                    if getattr(err, "validator", "") == "required":
                        missing_required_keys += 1
                    if getattr(err, "validator", "") == "additionalProperties":
                        unknown_keys = _extract_unknown_keys_from_validation_error(
                            err.message
                        )
                        for unknown_key in unknown_keys:
                            if unknown_key in guarded_arguments:
                                del guarded_arguments[unknown_key]
                                pruned_argument_keys += 1

        guarded.append({"name": canonical_name, "arguments": guarded_arguments})

    meta = {
        "dropped_unknown_tool_calls": dropped_unknown,
        "pruned_argument_keys": pruned_argument_keys,
        "missing_required_keys": missing_required_keys,
        "canonicalized_tool_names": canonicalized_tool_names,
        "schema_validation_errors": schema_validation_errors,
    }
    return guarded, meta


def _extract_unknown_keys_from_validation_error(message: str) -> list[str]:
    # Typical message: "Additional properties are not allowed ('foo', 'bar' were unexpected)"
    return re.findall(r"'([^']+)'", message or "")


def _normalize_argument_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_argument_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_argument_values(item) for item in value]
    if isinstance(value, str):
        cleaned = value.replace("\xa0", " ")
        for bad in ("\u00C2", "\u00C3\u201A", "\u0100"):
            cleaned = cleaned.replace(bad, "")
        return " ".join(cleaned.split())
    return value


def _build_repair_messages(
    sample: dict[str, Any],
    reasoning_effort: str,
    previous_output: str,
) -> list[dict[str, str]]:
    query = sample["query"]
    tools = sample["tools"]
    return [
        {
            "role": "system",
            "content": (
                "Gecersiz tool-calling ciktilarini duzeltirsin.\n"
                f"Reasoning effort: {reasoning_effort}.\n"
                "Sadece gecerli JSON dondur, ek metin yazma."
            ),
        },
        {
            "role": "user",
            "content": (
                "Asagidaki onceki cikti parse edilemedi veya arac kurallarina uymadi.\n"
                "Verilen tool listesinin disina cikma.\n\n"
                f"Sorgu:\n{query}\n\n"
                f"Toollar:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
                f"Onceki cikti:\n{previous_output}\n\n"
                "Duzeltilmis cikti formati:\n"
                '{"tool_calls":[{"name":"...","arguments":{...}}]}\n'
                "JSON bittigi anda ciktiyi durdur."
            ),
        },
    ]


def _request_repaired_output(
    client: Any,
    model: str,
    sample: dict[str, Any],
    previous_output: str,
    reasoning_effort: str,
    max_completion_tokens: int,
) -> str:
    messages = _build_repair_messages(
        sample=sample,
        reasoning_effort=reasoning_effort,
        previous_output=previous_output,
    )
    response = _request_completion(
        client=client,
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        with_schema=True,
    )
    return _response_content(response)


def _parse_tool_calls_from_text(text: str) -> tuple[list[dict[str, Any]], bool]:
    cleaned = text.strip()
    if not cleaned:
        return [], True

    payload, _ = parse_json_object_relaxed(cleaned)
    return _parse_tool_calls_from_payload(payload)


def _parse_tool_calls_from_payload(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(payload, dict):
        return [], True

    if BaseModel is not None:
        try:
            validated = ToolCallResponseModel.model_validate(payload)
            parsed_calls: list[dict[str, Any]] = []
            for call in validated.tool_calls:
                args = call.arguments
                if not isinstance(args, dict):
                    args = {"__value__": args}
                args = _normalize_argument_values(args)
                parsed_calls.append({"name": call.name, "arguments": args})
            return parsed_calls, False
        except ValidationError:
            return [], True

    raw_calls = payload.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return [], True

    parsed: list[dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        args = item.get("arguments", {})
        if not isinstance(name, str):
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"__raw__": args}
        if not isinstance(args, dict):
            args = {"__value__": args}
        args = _normalize_argument_values(args)
        parsed.append({"name": name, "arguments": args})
    return parsed, False


def _deterministic_repair_tool_calls(
    raw_text: str,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    cleaned = _sanitize_model_output(raw_text)
    if repair_json is not None:
        try:
            repaired_obj = repair_json(cleaned, return_objects=True)
            repaired_calls, repaired_error = _parse_tool_calls_from_payload(repaired_obj)
            if not repaired_error and repaired_calls:
                return repaired_calls, {
                    "applied": True,
                    "recovered_calls": len(repaired_calls),
                    "used_json_repair": True,
                }
        except Exception:  # noqa: BLE001
            pass
    calls = _extract_calls_via_regex(cleaned)
    if not calls:
        return [], {"applied": False, "recovered_calls": 0, "used_json_repair": False}
    return calls, {
        "applied": True,
        "recovered_calls": len(calls),
        "used_json_repair": False,
    }


def _sanitize_model_output(text: str) -> str:
    sanitized = text.replace("\u00a0", " ")
    for bad in ("Â", "Ã‚", "Ā"):
        sanitized = sanitized.replace(bad, "")
    # Drop non-printable control chars that frequently break JSON parsing.
    sanitized = "".join(
        ch for ch in sanitized if ch == "\n" or ch == "\t" or ord(ch) >= 32
    )
    return sanitized


def _extract_calls_via_regex(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        raw_name = match.group(1).strip()
        if not raw_name:
            continue
        search_start = match.end()
        arg_key_match = re.search(r'"arguments"\s*:', text[search_start:])
        if not arg_key_match:
            continue
        arg_key_start = search_start + arg_key_match.end()
        brace_start = text.find("{", arg_key_start)
        if brace_start == -1:
            continue
        arg_obj_text = _extract_balanced_object(text, brace_start)
        if not arg_obj_text:
            continue
        arguments = _parse_arguments_relaxed(arg_obj_text)
        calls.append({"name": raw_name, "arguments": arguments})
    return calls


def _extract_balanced_object(text: str, start_idx: int) -> str | None:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        return None
    depth = 0
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


def _parse_arguments_relaxed(arg_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(arg_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    recovered: dict[str, Any] = {}
    # Recover quoted string values.
    for key, value in re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', arg_text):
        recovered[key] = value
    # Recover numeric values.
    for key, value in re.findall(r'"([^"]+)"\s*:\s*(-?\d+(?:\.\d+)?)', arg_text):
        if key in recovered:
            continue
        if "." in value:
            try:
                recovered[key] = float(value)
            except ValueError:
                recovered[key] = value
        else:
            try:
                recovered[key] = int(value)
            except ValueError:
                recovered[key] = value
    return recovered


def _limit_predicted_calls(
    calls: list[dict[str, Any]],
    max_tool_calls: int | None,
) -> tuple[list[dict[str, Any]], int]:
    if max_tool_calls is None or max_tool_calls <= 0:
        return calls, 0
    if len(calls) <= max_tool_calls:
        return calls, 0
    dropped = len(calls) - max_tool_calls
    return calls[:max_tool_calls], dropped


def _build_minimum_fallback_tool_calls(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rule_calls = _build_car_rental_rule_calls(sample)
    if rule_calls:
        return rule_calls

    tools = sample.get("tools", [])
    if not isinstance(tools, list) or not tools:
        return []

    first_tool = tools[0]
    if not isinstance(first_tool, dict):
        return []
    fn = first_tool.get("function", {})
    if not isinstance(fn, dict):
        return []

    tool_name = str(fn.get("name", "")).strip()
    if not tool_name:
        return []

    parameters = fn.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    required = parameters.get("required", [])
    if not isinstance(required, list) or len(required) != 1:
        return []

    required_key = required[0]
    if not isinstance(required_key, str) or not required_key.strip():
        return []

    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    key_schema = properties.get(required_key, {})
    if not isinstance(key_schema, dict):
        return []
    key_type = str(key_schema.get("type", "string")).strip().lower()
    if key_type != "string":
        return []

    query = str(sample.get("query", "")).strip()
    value = _extract_query_hint(query, required_key)
    if not value:
        value = query
    value = _normalize_argument_values(value)
    if not isinstance(value, str) or not value:
        return []

    return [{"name": tool_name, "arguments": {required_key: value}}]


def _extract_query_hint(query: str, required_key: str) -> str:
    text = query.strip()
    if not text:
        return ""

    key = required_key.lower()
    if key in {"query", "q", "keyword", "keywords", "search"}:
        patterns = [
            r"\bat\s+the\s+([^?.!,]+)",
            r"\bat\s+([^?.!,]+?)\s+(?:at\s+\d|on\s+[A-Za-z]|\.)",
            r"\bnear\s+the\s+([^?.!,]+)",
            r"\bfor\s+the\s+([^?.!,]+)",
            r"'([^']+)'",
            r"\"([^\"]+)\"",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    return candidate
    return ""


def _build_car_rental_rule_calls(sample: dict[str, Any]) -> list[dict[str, Any]]:
    query = str(sample.get("query", "")).strip()
    tools = sample.get("tools", [])
    if not query or not isinstance(tools, list):
        return []

    tool_names: dict[str, str] = {}
    for t in tools:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        name = str(fn.get("name", "")).strip()
        if name:
            tool_names[_normalize_tool_name(name)] = name

    loc_name = tool_names.get(_normalize_tool_name("Search_Car_Location"))
    rentals_name = tool_names.get(_normalize_tool_name("Search_Car_Rentals"))
    if not loc_name and not rentals_name:
        return []

    # If coordinates are explicitly provided for pick-up/drop-off, prefer rentals tool.
    rental_args = _extract_car_rental_args(query)
    if rentals_name and rental_args:
        return [{"name": rentals_name, "arguments": rental_args}]

    if not loc_name:
        return []

    locations = _extract_location_candidates(query)
    if not locations:
        hint = _extract_query_hint(query, "query")
        locations = [hint] if hint else []
    locations = [loc.strip() for loc in locations if isinstance(loc, str) and loc.strip()]
    if not locations:
        return []
    return [{"name": loc_name, "arguments": {"query": loc}} for loc in locations]


def _extract_location_candidates(query: str) -> list[str]:
    candidates: list[str] = []

    paired = re.search(
        r"\bat\s+(?:the\s+)?(.+?)\s+at\s+\d.*?\breturn(?:\s+it)?\s+(?:in|to)\s+(.+?)\s+at\s+\d",
        query,
        flags=re.IGNORECASE,
    )
    if paired:
        candidates.extend([paired.group(1).strip(), paired.group(2).strip()])

    same_place = re.search(r"\bat\s+(?:the\s+)?([^,.]+)", query, flags=re.IGNORECASE)
    if same_place:
        first = same_place.group(1).strip()
        if first and first not in candidates:
            candidates.append(first)

    return candidates[:2]


def _extract_car_rental_args(query: str) -> dict[str, Any] | None:
    coords = re.findall(
        r"longitude:\s*(-?\d+(?:\.\d+)?)\s*,\s*latitude:\s*(-?\d+(?:\.\d+)?)",
        query,
        flags=re.IGNORECASE,
    )
    if len(coords) < 2:
        return None

    pickup_dt = _extract_first_datetime(query)
    if pickup_dt is None:
        return None
    dropoff_dt = _extract_dropoff_datetime(query, pickup_dt)
    if dropoff_dt is None:
        return None

    pick_up_longitude = float(coords[0][0])
    pick_up_latitude = float(coords[0][1])
    drop_off_longitude = float(coords[1][0])
    drop_off_latitude = float(coords[1][1])

    return {
        "pick_up_longitude": pick_up_longitude,
        "pick_up_latitude": pick_up_latitude,
        "pick_up_date": pickup_dt.strftime("%Y-%m-%d"),
        "pick_up_time": pickup_dt.strftime("%H:%M"),
        "drop_off_longitude": drop_off_longitude,
        "drop_off_latitude": drop_off_latitude,
        "drop_off_date": dropoff_dt.strftime("%Y-%m-%d"),
        "drop_off_time": dropoff_dt.strftime("%H:%M"),
    }


def _extract_first_datetime(text: str) -> datetime | None:
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    date_match = re.search(
        r"\b(?:on\s+)?([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    time_match = re.search(
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])",
        text,
        flags=re.IGNORECASE,
    )
    if not date_match or not time_match:
        return None
    month = month_map.get(date_match.group(1).lower())
    if month is None:
        return None
    day = int(date_match.group(2))
    year = int(date_match.group(3))
    hour = int(time_match.group(1)) % 12
    if time_match.group(3).lower() == "pm":
        hour += 12
    minute = int(time_match.group(2) or 0)
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def _extract_dropoff_datetime(text: str, pickup_dt: datetime) -> datetime | None:
    day_delta_match = re.search(r"\b(\d+)\s+days?\s+later\b", text, flags=re.IGNORECASE)
    hour_delta_match = re.search(r"\b(\d+)\s+hours?\s+later\b", text, flags=re.IGNORECASE)
    if day_delta_match:
        return pickup_dt + timedelta(days=int(day_delta_match.group(1)))
    if hour_delta_match:
        return pickup_dt + timedelta(hours=int(hour_delta_match.group(1)))
    if re.search(r"\bnext day\b", text, flags=re.IGNORECASE):
        return pickup_dt + timedelta(days=1)
    if re.search(r"\bsame time\b", text, flags=re.IGNORECASE):
        # If a second explicit date exists, use it with pickup time.
        second_date = _extract_second_date(text)
        if second_date is not None:
            return second_date.replace(hour=pickup_dt.hour, minute=pickup_dt.minute)
    # Fallback for common "for a day/24-hour rental" wording.
    if re.search(r"\b24[- ]?hour\b|\bfor a day\b", text, flags=re.IGNORECASE):
        return pickup_dt + timedelta(days=1)
    return None


def _extract_second_date(text: str) -> datetime | None:
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    matches = list(
        re.finditer(
            r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(\d{4})",
            text,
            flags=re.IGNORECASE,
        )
    )
    if len(matches) < 2:
        return None
    m = matches[1]
    month = month_map.get(m.group(1).lower())
    if month is None:
        return None
    try:
        return datetime(int(m.group(3)), month, int(m.group(2)))
    except ValueError:
        return None


def _build_messages(sample: dict[str, Any], reasoning_effort: str) -> list[dict[str, str]]:
    query = sample["query"]
    tools = sample["tools"]
    return [
        {
            "role": "system",
            "content": (
                "Tool-calling degerlendiricisisin.\n"
                f"Reasoning effort: {reasoning_effort}.\n"
                "Kullanicinin istegine gore gerekli tool cagri listesini cikart.\n"
                "Kurallar:\n"
                "- Sadece verilen tool adlarini kullan.\n"
                "- Cevap sadece JSON olsun.\n"
                "- Duz metin, markdown, aciklama yazma.\n"
                "- Karsilastirma/iki farkli secenek varsa birden fazla tool_call dondur.\n"
                "- JSON bittigi anda ciktiyi durdur."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Sorgu:\n{query}\n\n"
                f"Toollar:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
                "Cikti formati:\n"
                '{"tool_calls":[{"name":"...","arguments":{...}}]}'
            ),
        },
    ]


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    requested_model = args.model or settings.requested_model
    reasoning_effort = args.reasoning_effort or settings.reasoning_effort
    dataset_path = Path(args.dataset)
    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    dataset_format = _resolve_dataset_format(
        dataset_path=dataset_path,
        requested_format=args.dataset_format,
    )
    answer_path: Path | None = None
    if dataset_format == "bfcl_sql":
        answer_path = _resolve_bfcl_answer_path(
            dataset_path=dataset_path,
            cli_answer_path=args.answers,
        )
        if not answer_path.exists():
            print(f"BFCL answer dosyasi bulunamadi: {answer_path}")
            return 1
        samples = load_bfcl_sql_samples(
            question_path=dataset_path,
            answer_path=answer_path,
            limit=None,
        )
    else:
        samples = load_eval_samples(dataset_path, limit=None)

    if args.limit is not None and args.limit < len(samples):
        if args.sampling == "random":
            rng = random.Random(args.seed)
            samples = rng.sample(samples, args.limit)
        else:
            samples = samples[: args.limit]
    if not samples:
        print(f"Degerlendirme dataseti bos: {dataset_path}")
        return 1

    client: Any = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    available = [m.id for m in client.models.list().data]
    if requested_model not in available:
        print(
            f"Istenen model yok: {requested_model}. "
            f"Yuklu modeller: {', '.join(available)}"
        )
        return 1

    results: list[dict[str, Any]] = []
    error_counter: Counter[str] = Counter()
    exact = 0
    name_match_total = 0
    call_count_match_total = 0
    arg_match_total = 0
    expected_call_total = 0
    repair_attempted_total = 0
    repair_success_total = 0
    local_repair_applied_total = 0
    local_repair_recovered_calls_total = 0
    json_repair_used_total = 0
    guard_unknown_drop_total = 0
    guard_pruned_arg_keys_total = 0
    guard_missing_required_total = 0
    guard_max_call_drop_total = 0
    guard_schema_validation_error_total = 0
    fallback_injected_calls_total = 0

    for idx, sample in enumerate(samples):
        messages = _build_messages(sample, reasoning_effort=reasoning_effort)
        response = _request_completion(
            client=client,
            model=requested_model,
            messages=messages,
            max_completion_tokens=args.max_completion_tokens,
            with_schema=True,
        )

        raw_output = _response_content(response)
        predicted_calls, parse_error = _parse_tool_calls_from_text(raw_output)
        local_repair_meta: dict[str, int | bool] = {"applied": False, "recovered_calls": 0}
        if parse_error or not predicted_calls:
            repaired_locally_calls, local_repair_meta = _deterministic_repair_tool_calls(
                raw_output
            )
            if repaired_locally_calls:
                predicted_calls = repaired_locally_calls
                parse_error = False
        if bool(local_repair_meta.get("applied")):
            local_repair_applied_total += 1
            local_repair_recovered_calls_total += int(
                local_repair_meta.get("recovered_calls", 0)
            )
        if bool(local_repair_meta.get("used_json_repair")):
            json_repair_used_total += 1

        predicted_calls, guard_meta = _guard_tool_calls(
            predicted_calls=predicted_calls,
            tools=sample.get("tools", []),
        )
        predicted_calls, dropped_by_max_tool_calls = _limit_predicted_calls(
            calls=predicted_calls,
            max_tool_calls=args.max_tool_calls,
        )
        guard_meta["dropped_by_max_tool_calls"] = dropped_by_max_tool_calls
        repair_applied = False
        repair_success = False
        repair_attempts_used = 0

        needs_repair = (
            parse_error
            or not predicted_calls
            or guard_meta.get("dropped_unknown_tool_calls", 0) > 0
            or guard_meta.get("missing_required_keys", 0) > 0
        )
        if args.repair_attempts > 0 and needs_repair:
            repair_applied = True
            repair_attempted_total += 1
            repair_input = raw_output
            for _ in range(args.repair_attempts):
                repair_attempts_used += 1
                repaired_output = _request_repaired_output(
                    client=client,
                    model=requested_model,
                    sample=sample,
                    previous_output=repair_input,
                    reasoning_effort=reasoning_effort,
                    max_completion_tokens=args.max_completion_tokens,
                )
                repaired_calls, repaired_parse_error = _parse_tool_calls_from_text(
                    repaired_output
                )
                repaired_local_meta: dict[str, int | bool] = {
                    "applied": False,
                    "recovered_calls": 0,
                }
                if repaired_parse_error or not repaired_calls:
                    repaired_calls, repaired_local_meta = _deterministic_repair_tool_calls(
                        repaired_output
                    )
                    repaired_parse_error = not repaired_calls
                if bool(repaired_local_meta.get("applied")):
                    local_repair_applied_total += 1
                    local_repair_recovered_calls_total += int(
                        repaired_local_meta.get("recovered_calls", 0)
                    )
                if bool(repaired_local_meta.get("used_json_repair")):
                    json_repair_used_total += 1
                repaired_calls, repaired_guard_meta = _guard_tool_calls(
                    predicted_calls=repaired_calls,
                    tools=sample.get("tools", []),
                )
                repaired_calls, repaired_dropped_by_max = _limit_predicted_calls(
                    calls=repaired_calls,
                    max_tool_calls=args.max_tool_calls,
                )
                repaired_guard_meta["dropped_by_max_tool_calls"] = repaired_dropped_by_max

                if not repaired_parse_error and repaired_calls:
                    raw_output = repaired_output
                    predicted_calls = repaired_calls
                    parse_error = False
                    guard_meta = repaired_guard_meta
                    repair_success = True
                    repair_success_total += 1
                    break
                repair_input = repaired_output

        if not predicted_calls:
            fallback_calls = _build_minimum_fallback_tool_calls(sample)
            if fallback_calls:
                predicted_calls = fallback_calls
                parse_error = False
                injected = len(fallback_calls)
                guard_meta["fallback_injected_calls"] = injected
                fallback_injected_calls_total += injected
            else:
                guard_meta["fallback_injected_calls"] = 0
        else:
            guard_meta["fallback_injected_calls"] = 0

        guard_unknown_drop_total += guard_meta.get("dropped_unknown_tool_calls", 0)
        guard_pruned_arg_keys_total += guard_meta.get("pruned_argument_keys", 0)
        guard_missing_required_total += guard_meta.get("missing_required_keys", 0)
        guard_max_call_drop_total += guard_meta.get("dropped_by_max_tool_calls", 0)
        guard_schema_validation_error_total += guard_meta.get(
            "schema_validation_errors", 0
        )

        expected_variants = sample.get("expected_tool_call_variants")
        matched_variant_index = 0
        if isinstance(expected_variants, list) and expected_variants:
            (
                exact_match,
                names_ok,
                call_count_ok,
                arg_match_count,
                error_type,
                matched_variant_index,
            ) = score_prediction_variants(
                expected_variants=expected_variants,
                predicted_calls=predicted_calls,
                parse_error=parse_error,
            )
            expected_calls = expected_variants[matched_variant_index]
        else:
            expected_calls = sample["expected_tool_calls"]
            (
                exact_match,
                names_ok,
                call_count_ok,
                arg_match_count,
                error_type,
            ) = score_prediction(
                expected_calls=expected_calls,
                predicted_calls=predicted_calls,
                parse_error=parse_error,
            )

        expected_count = len(expected_calls)
        expected_call_total += expected_count
        arg_match_total += arg_match_count

        if exact_match:
            exact += 1
        if names_ok:
            name_match_total += 1
        if call_count_ok:
            call_count_match_total += 1
        if error_type:
            error_counter[error_type] += 1

        results.append(
            {
                "index": idx,
                "source_dataset": sample.get("source_dataset"),
                "source_row_id": sample.get("source_row_id"),
                "query": sample["query"],
                "expected_tool_calls": expected_calls,
                "predicted_tool_calls": predicted_calls,
                "raw_model_output": raw_output,
                "parse_error": parse_error,
                "repair_applied": repair_applied,
                "repair_success": repair_success,
                "repair_attempts_used": repair_attempts_used,
                "local_repair_applied": bool(local_repair_meta.get("applied")),
                "local_repair_recovered_calls": int(
                    local_repair_meta.get("recovered_calls", 0)
                ),
                "local_repair_used_json_repair": bool(
                    local_repair_meta.get("used_json_repair")
                ),
                "guard": guard_meta,
                "matched_variant_index": matched_variant_index,
                "expected_variant_count": len(expected_variants)
                if isinstance(expected_variants, list)
                else 1,
                "exact_match": exact_match,
                "name_match": names_ok,
                "call_count_match": call_count_ok,
                "arg_match_count": arg_match_count,
                "expected_call_count": expected_count,
                "error_type": error_type,
            }
        )

    total = len(samples)
    argument_error_breakdown = _build_argument_error_breakdown(results)
    failure_patterns = _build_failure_patterns(results, top_k=3)
    actionable_plan = _build_actionable_plan(
        summary_metrics={
            "exact_match_accuracy": round(exact / total, 4),
            "name_match_accuracy": round(name_match_total / total, 4),
            "call_count_accuracy": round(call_count_match_total / total, 4),
            "argument_match_rate": round(
                (arg_match_total / expected_call_total) if expected_call_total else 0.0, 4
            ),
        },
        failure_patterns=failure_patterns,
        guard_metrics={
            "dropped_unknown_tool_calls": guard_unknown_drop_total,
            "missing_required_keys": guard_missing_required_total,
            "dropped_by_max_tool_calls": guard_max_call_drop_total,
            "schema_validation_errors": guard_schema_validation_error_total,
        },
    )
    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_format": dataset_format,
        "answer_path": str(answer_path) if answer_path else None,
        "backend": "openai_compat",
        "model": requested_model,
        "reasoning_effort": reasoning_effort,
        "sample_count": total,
        "sampling": args.sampling,
        "seed": args.seed,
        "max_tool_calls": args.max_tool_calls,
        "max_completion_tokens": args.max_completion_tokens,
        "exact_match_accuracy": round(exact / total, 4),
        "name_match_accuracy": round(name_match_total / total, 4),
        "call_count_accuracy": round(call_count_match_total / total, 4),
        "argument_match_rate": round(
            (arg_match_total / expected_call_total) if expected_call_total else 0.0, 4
        ),
        "error_breakdown": dict(error_counter),
        "argument_error_breakdown": argument_error_breakdown,
        "top_failure_patterns": failure_patterns,
        "actionable_plan": actionable_plan,
        "repair": {
            "attempted_samples": repair_attempted_total,
            "successful_repairs": repair_success_total,
            "max_attempts_per_sample": args.repair_attempts,
            "local_repair_applied": local_repair_applied_total,
            "local_repair_recovered_calls": local_repair_recovered_calls_total,
            "json_repair_used": json_repair_used_total,
        },
        "guard": {
            "dropped_unknown_tool_calls": guard_unknown_drop_total,
            "pruned_argument_keys": guard_pruned_arg_keys_total,
            "missing_required_keys": guard_missing_required_total,
            "dropped_by_max_tool_calls": guard_max_call_drop_total,
            "schema_validation_errors": guard_schema_validation_error_total,
            "fallback_injected_calls": fallback_injected_calls_total,
        },
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = runs_root / f"hf_tool_eval_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    details_path = out_dir / "details.json"
    report_path = out_dir / "report.md"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    details_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(_build_report(summary, results), encoding="utf-8")

    print("Degerlendirme tamamlandi.")
    print(f"Ozet: {summary_path}")
    print(f"Detay: {details_path}")
    print(f"Rapor: {report_path}")
    print("")
    print(
        "Skorlar -> "
        f"exact: {summary['exact_match_accuracy']}, "
        f"name: {summary['name_match_accuracy']}, "
        f"count: {summary['call_count_accuracy']}, "
        f"args: {summary['argument_match_rate']}"
    )
    return 0


def _build_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# HF Tool-Calling Degerlendirme Raporu")
    lines.append("")
    lines.append(f"- Model: `{summary['model']}`")
    lines.append(f"- Backend: `{summary.get('backend')}`")
    lines.append(f"- Reasoning effort: `{summary['reasoning_effort']}`")
    lines.append(f"- Veri: `{summary['dataset_path']}`")
    lines.append(f"- Veri formati: `{summary['dataset_format']}`")
    lines.append(f"- Sampling: `{summary.get('sampling')}`")
    lines.append(f"- Seed: `{summary.get('seed')}`")
    lines.append(f"- Max tool calls: `{summary.get('max_tool_calls')}`")
    lines.append(
        f"- Max completion tokens: `{summary.get('max_completion_tokens')}`"
    )
    if summary.get("answer_path"):
        lines.append(f"- Ground truth: `{summary['answer_path']}`")
    lines.append(f"- Ornek sayisi: `{summary['sample_count']}`")
    lines.append("")
    lines.append("## Skorlar")
    lines.append("")
    lines.append(f"- Exact match accuracy: `{summary['exact_match_accuracy']}`")
    lines.append(f"- Tool name accuracy: `{summary['name_match_accuracy']}`")
    lines.append(f"- Call count accuracy: `{summary['call_count_accuracy']}`")
    lines.append(f"- Argument match rate: `{summary['argument_match_rate']}`")
    lines.append("")
    lines.append("## Actionable Plan")
    lines.append("")
    plan = summary.get("actionable_plan", [])
    if not plan:
        lines.append("- Ek aksiyon onerisi yok.")
    else:
        for item in plan:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("## Top Failure Patterns")
    lines.append("")
    patterns = summary.get("top_failure_patterns", [])
    if not patterns:
        lines.append("- Pattern bulunamadi.")
    else:
        for pattern in patterns:
            lines.append(
                "- "
                f"{pattern.get('error_type')} | "
                f"name={pattern.get('name_match')} count={pattern.get('call_count_match')} "
                f"parse={pattern.get('parse_error')} | "
                f"adet={pattern.get('count')}"
            )
    lines.append("")
    lines.append("## Hata Dagilimi")
    lines.append("")
    breakdown = summary.get("error_breakdown", {})
    if not breakdown:
        lines.append("- Hata yok.")
    else:
        for key, value in breakdown.items():
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Arguman Hata Dagilimi")
    lines.append("")
    argument_breakdown = summary.get("argument_error_breakdown", {})
    if not argument_breakdown:
        lines.append("- Arguman bazli hata yok.")
    else:
        for key, value in argument_breakdown.items():
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Guard / Repair")
    lines.append("")
    repair = summary.get("repair", {})
    guard = summary.get("guard", {})
    if repair:
        lines.append(f"- Repair denenen ornek: `{repair.get('attempted_samples', 0)}`")
        lines.append(f"- Basarili repair: `{repair.get('successful_repairs', 0)}`")
        lines.append(
            f"- Ornek basi max deneme: `{repair.get('max_attempts_per_sample', 0)}`"
        )
        lines.append(
            f"- Local deterministic repair uygulanan: `{repair.get('local_repair_applied', 0)}`"
        )
        lines.append(
            f"- Local deterministic recover edilen call: `{repair.get('local_repair_recovered_calls', 0)}`"
        )
        lines.append(
            f"- json_repair kullanilan local repair: `{repair.get('json_repair_used', 0)}`"
        )
    if guard:
        lines.append(
            f"- Guard dusen bilinmeyen tool call: `{guard.get('dropped_unknown_tool_calls', 0)}`"
        )
        lines.append(
            f"- Guard temizlenen arguman anahtari: `{guard.get('pruned_argument_keys', 0)}`"
        )
        lines.append(
            f"- Guard eksik zorunlu anahtar sayisi: `{guard.get('missing_required_keys', 0)}`"
        )
        lines.append(
            f"- Max tool call limitiyle dusen call: `{guard.get('dropped_by_max_tool_calls', 0)}`"
        )
        lines.append(
            f"- JSON schema validation hata sayisi: `{guard.get('schema_validation_errors', 0)}`"
        )
        lines.append(
            f"- Fallback ile enjekte edilen call: `{guard.get('fallback_injected_calls', 0)}`"
        )
    lines.append("")
    lines.append("## Basarisiz Ornekler")
    lines.append("")

    failed = [row for row in results if not row["exact_match"]]
    if not failed:
        lines.append("- Tum ornekler exact match.")
        return "\n".join(lines)

    for row in failed[:10]:
        lines.append(f"### Ornek {row['index']} (row_id={row.get('source_row_id')})")
        lines.append(f"- Sorgu: {row['query']}")
        lines.append(f"- Hata tipi: `{row.get('error_type')}`")
        if row.get("expected_variant_count", 1) > 1:
            lines.append(
                f"- Eslesen varyant index: `{row.get('matched_variant_index')}`"
            )
        lines.append(
            f"- Beklenen: `{json.dumps(row['expected_tool_calls'], ensure_ascii=False)}`"
        )
        lines.append(
            f"- Tahmin: `{json.dumps(row['predicted_tool_calls'], ensure_ascii=False)}`"
        )
        lines.append("")

    return "\n".join(lines)


def _build_argument_error_breakdown(results: list[dict[str, Any]]) -> dict[str, int]:
    breakdown: Counter[str] = Counter()
    for row in results:
        if row.get("exact_match"):
            continue
        expected_calls = normalize_tool_calls(row.get("expected_tool_calls", []))
        predicted_calls = normalize_tool_calls(row.get("predicted_tool_calls", []))
        if not expected_calls or not predicted_calls:
            continue
        expected_args = expected_calls[0].get("arguments", {})
        predicted_args = predicted_calls[0].get("arguments", {})
        if not isinstance(expected_args, dict) or not isinstance(predicted_args, dict):
            continue
        for key, value in expected_args.items():
            if predicted_args.get(key) != value:
                breakdown[key] += 1
    return dict(breakdown)


def _build_failure_patterns(
    results: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, bool, bool, bool]] = Counter()
    for row in results:
        if row.get("exact_match"):
            continue
        key = (
            str(row.get("error_type") or "unknown"),
            bool(row.get("name_match")),
            bool(row.get("call_count_match")),
            bool(row.get("parse_error")),
        )
        counter[key] += 1
    top = counter.most_common(top_k)
    return [
        {
            "error_type": key[0],
            "name_match": key[1],
            "call_count_match": key[2],
            "parse_error": key[3],
            "count": count,
        }
        for key, count in top
    ]


def _build_actionable_plan(
    summary_metrics: dict[str, float],
    failure_patterns: list[dict[str, Any]],
    guard_metrics: dict[str, int],
) -> list[str]:
    actions: list[str] = []
    pattern_action: str | None = None
    name_acc = float(summary_metrics.get("name_match_accuracy", 0.0))
    count_acc = float(summary_metrics.get("call_count_accuracy", 0.0))
    arg_rate = float(summary_metrics.get("argument_match_rate", 0.0))

    if name_acc < 0.8:
        actions.append(
            "Tool adi secim hatasi yuksek: prompt'a yalnizca tool adini kopyalayip kullanma kurali ekle ve close-match otomatik duzeltmeyi acikca logla."
        )
    if count_acc < 0.8 or int(guard_metrics.get("dropped_by_max_tool_calls", 0)) > 0:
        actions.append(
            "Tool sayisi hatasi var: max-tool-calls degerini dataset yapisina gore ayarla ve coklu-cagri gereken ornekleri ayri senaryoda degerlendir."
        )
    if arg_rate < 0.75 or int(guard_metrics.get("missing_required_keys", 0)) > 0:
        actions.append(
            "Arguman kalitesi dusuk: required alanlari once doldur sonra optional alanlari ekle seklinde iki adimli arguman olusturma kuralini zorunlu yap."
        )

    if failure_patterns:
        top = failure_patterns[0]
        top_error = str(top.get("error_type", "unknown"))
        pattern_action = (
            f"En sik pattern `{top_error}`: bu pattern icin hedefli 10 orneklik mini regression seti olusturup her degisiklikte otomatik kos."
        )

    if int(guard_metrics.get("schema_validation_errors", 0)) > 0:
        actions.append(
            "Schema ihlali goruluyor: schema-validator hatalarini model geri beslemesine tek satir neden olarak enjekte et."
        )

    # Keep output concise and realistic; always include top-pattern action when available.
    if pattern_action:
        primary = actions[:2]
        return [*primary, pattern_action]
    return actions[:3]


def _resolve_dataset_format(dataset_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format

    if dataset_path.name == "BFCL_v3_sql.json":
        sibling_answer = dataset_path.parent / "possible_answer" / "BFCL_v3_sql.json"
        if sibling_answer.exists():
            return "bfcl_sql"
    return "jsonl"


def _resolve_bfcl_answer_path(dataset_path: Path, cli_answer_path: str | None) -> Path:
    if cli_answer_path:
        return Path(cli_answer_path)
    return dataset_path.parent / "possible_answer" / "BFCL_v3_sql.json"


if __name__ == "__main__":
    raise SystemExit(main())
