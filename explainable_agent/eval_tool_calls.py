from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SampleResult:
    index: int
    source_row_id: int | str | None
    query: str
    expected_tool_calls: list[dict[str, Any]]
    predicted_tool_calls: list[dict[str, Any]]
    raw_model_output: str
    parse_error: bool
    exact_match: bool
    name_match: bool
    call_count_match: bool
    arg_match_count: int
    expected_call_count: int
    error_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_row_id": self.source_row_id,
            "query": self.query,
            "expected_tool_calls": self.expected_tool_calls,
            "predicted_tool_calls": self.predicted_tool_calls,
            "raw_model_output": self.raw_model_output,
            "parse_error": self.parse_error,
            "exact_match": self.exact_match,
            "name_match": self.name_match,
            "call_count_match": self.call_count_match,
            "arg_match_count": self.arg_match_count,
            "expected_call_count": self.expected_call_count,
            "error_type": self.error_type,
        }


BFCL_SQL_ARGUMENT_KEYS = {
    "sql_keyword",
    "table_name",
    "columns",
    "insert_values",
    "update_values",
    "conditions",
}


def load_eval_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = _read_jsonl_rows(path)
    if limit is not None:
        return rows[:limit]
    return rows


def load_bfcl_sql_samples(
    question_path: Path,
    answer_path: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    question_rows = _read_jsonl_rows(question_path)
    answer_rows = _read_jsonl_rows(answer_path)
    answers_by_id = {
        str(row.get("id", "")).strip(): row
        for row in answer_rows
        if str(row.get("id", "")).strip()
    }

    rows: list[dict[str, Any]] = []
    for question_row in question_rows:
        row_id = str(question_row.get("id", "")).strip()
        if not row_id:
            continue
        answer_row = answers_by_id.get(row_id)
        if not answer_row:
            continue

        tools = _normalize_bfcl_tools(question_row.get("function", []))
        variants = _normalize_bfcl_ground_truth(answer_row.get("ground_truth", []))
        if not variants:
            variants = [[]]

        rows.append(
            {
                "source_dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                "source_subset": "BFCL_v3_sql",
                "source_row_id": row_id,
                "query": _extract_query_text(question_row.get("question", [])),
                "tools": tools,
                "expected_tool_calls": variants[0],
                "expected_tool_call_variants": variants,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, str):
        cleaned = value.replace("\xa0", " ").replace("Â", "")
        return " ".join(cleaned.split())
    return value


def normalize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in calls:
        name = str(call.get("name", "")).strip()
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__raw__": arguments}
        if not isinstance(arguments, dict):
            arguments = {"__value__": arguments}
        if _looks_like_bfcl_sql_arguments(arguments):
            arguments = _normalize_bfcl_sql_arguments(arguments)
        else:
            arguments = normalize_value(arguments)
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def score_prediction(
    expected_calls: list[dict[str, Any]],
    predicted_calls: list[dict[str, Any]],
    parse_error: bool,
) -> tuple[bool, bool, bool, int, str | None]:
    expected = normalize_tool_calls(expected_calls)
    predicted = normalize_tool_calls(predicted_calls)

    expected_count = len(expected)
    call_count_match = expected_count == len(predicted)

    expected_names = [c["name"] for c in expected]
    predicted_names = [c["name"] for c in predicted]
    name_match = expected_names == predicted_names

    arg_match_count = 0
    for exp, pred in zip(expected, predicted):
        if _arguments_equal(exp["arguments"], pred["arguments"]):
            arg_match_count += 1

    exact_match = call_count_match and name_match and (arg_match_count == expected_count)

    error_type: str | None = None
    if parse_error:
        error_type = "parse_error"
    elif not call_count_match:
        error_type = "call_count_mismatch"
    elif not name_match:
        error_type = "wrong_tool_name"
    elif arg_match_count != expected_count:
        error_type = "wrong_arguments"

    return exact_match, name_match, call_count_match, arg_match_count, error_type


def score_prediction_variants(
    expected_variants: list[list[dict[str, Any]]],
    predicted_calls: list[dict[str, Any]],
    parse_error: bool,
) -> tuple[bool, bool, bool, int, str | None, int]:
    variants = expected_variants or [[]]

    best_index = 0
    best_result = score_prediction(
        expected_calls=variants[0],
        predicted_calls=predicted_calls,
        parse_error=parse_error,
    )
    best_rank = _score_rank(best_result)

    for index, expected_calls in enumerate(variants[1:], start=1):
        result = score_prediction(
            expected_calls=expected_calls,
            predicted_calls=predicted_calls,
            parse_error=parse_error,
        )
        rank = _score_rank(result)
        if rank > best_rank:
            best_rank = rank
            best_result = result
            best_index = index

    return (
        best_result[0],
        best_result[1],
        best_result[2],
        best_result[3],
        best_result[4],
        best_index,
    )


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _extract_query_text(question_messages: Any) -> str:
    if not isinstance(question_messages, list):
        return str(question_messages or "")
    parts: list[str] = []
    for message in question_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        if role != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts)


def _normalize_bfcl_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []
    tools: list[dict[str, Any]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        name = str(raw_tool.get("name", "")).strip()
        if not name:
            continue
        description = str(raw_tool.get("description", "")).strip()
        parameters = _normalize_schema(raw_tool.get("parameters", {}))
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return tools


def _normalize_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        normalized: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "type" and value == "dict":
                normalized[key] = "object"
            else:
                normalized[key] = _normalize_schema(value)
        return normalized
    if isinstance(schema, list):
        return [_normalize_schema(item) for item in schema]
    return schema


def _normalize_bfcl_ground_truth(ground_truth: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(ground_truth, list):
        return []

    variants: list[list[dict[str, Any]]] = []
    for option in ground_truth:
        if not isinstance(option, dict):
            continue
        calls: list[dict[str, Any]] = []
        for tool_name, raw_arguments in option.items():
            if not isinstance(raw_arguments, dict):
                raw_arguments = {"__value__": raw_arguments}
            calls.append(
                {
                    "name": str(tool_name).strip(),
                    "arguments": _normalize_bfcl_sql_arguments(raw_arguments),
                }
            )
        if calls:
            variants.append(calls)
    return variants


def _looks_like_bfcl_sql_arguments(arguments: dict[str, Any]) -> bool:
    if not arguments:
        return False
    return set(arguments.keys()).issubset(BFCL_SQL_ARGUMENT_KEYS)


def _normalize_bfcl_sql_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    if "sql_keyword" in arguments:
        keyword = _normalize_scalar(_first_value(arguments.get("sql_keyword")))
        if isinstance(keyword, str):
            keyword = _normalize_sql_keyword(keyword)
        normalized["sql_keyword"] = keyword

    if "table_name" in arguments:
        table_name = _normalize_scalar(_first_value(arguments.get("table_name")))
        if isinstance(table_name, str):
            table_name = _normalize_identifier(table_name)
        normalized["table_name"] = table_name

    if "columns" in arguments:
        columns = _normalize_identifier_list(arguments.get("columns"))
        if columns:
            normalized["columns"] = columns

    if "conditions" in arguments:
        conditions = _normalize_condition_list(arguments.get("conditions"))
        if conditions:
            normalized["conditions"] = conditions

    if "update_values" in arguments:
        update_values = _normalize_scalar_list(arguments.get("update_values"))
        if update_values:
            normalized["update_values"] = update_values

    if "insert_values" in arguments:
        insert_values = _normalize_insert_rows(arguments.get("insert_values"))
        if insert_values:
            normalized["insert_values"] = insert_values

    for key, value in arguments.items():
        if key not in normalized:
            normalized[key] = normalize_value(value)
    return normalized


def _first_value(value: Any) -> Any:
    current = value
    while isinstance(current, list) and len(current) == 1:
        current = current[0]
    return current


def _normalize_identifier(text: str) -> str:
    trimmed = text.replace("\xa0", " ").strip()
    trimmed = trimmed.strip('"')
    trimmed = trimmed.strip("'")
    trimmed = trimmed.strip("`")
    compact = re.sub(r"[^a-zA-Z0-9*]", "", trimmed.lower())
    return compact


def _normalize_scalar(value: Any) -> Any:
    scalar = _first_value(value)
    if isinstance(scalar, str):
        stripped = scalar.replace("\xa0", " ").replace("Â", "").strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                return stripped
        if re.fullmatch(r"[+-]?\d+\.\d+", stripped):
            try:
                return float(stripped)
            except ValueError:
                return stripped
        return stripped
    return normalize_value(scalar)


def _normalize_identifier_list(value: Any) -> list[str]:
    if value is None:
        return []
    current = value
    while isinstance(current, list) and len(current) == 1 and isinstance(current[0], list):
        current = current[0]
    if not isinstance(current, list):
        current = [current]
    identifiers: list[str] = []
    for item in current:
        scalar = _normalize_scalar(item)
        if isinstance(scalar, str):
            identifier = _normalize_identifier(scalar)
            if identifier:
                identifiers.append(identifier)
        else:
            text = str(scalar).strip()
            if text:
                identifiers.append(text)
    return identifiers


def _normalize_condition_list(value: Any) -> list[str]:
    if value is None:
        return []
    current = value
    while isinstance(current, list) and len(current) == 1 and isinstance(current[0], list):
        current = current[0]
    if not isinstance(current, list):
        current = [current]
    conditions: list[str] = []
    for item in current:
        scalar = _normalize_scalar(item)
        if isinstance(scalar, str):
            parsed = _normalize_condition_scalar(scalar)
            if parsed:
                conditions.append(parsed)
        else:
            text = str(scalar).strip()
            if text:
                conditions.append(text)
    return conditions


def _normalize_scalar_list(value: Any) -> list[Any]:
    if value is None:
        return []
    current = value
    while isinstance(current, list) and len(current) == 1 and isinstance(current[0], list):
        current = current[0]
    if not isinstance(current, list):
        current = [current]
    normalized = [normalize_value(_normalize_scalar(item)) for item in current]
    filtered: list[Any] = []
    for item in normalized:
        if item is None:
            continue
        if isinstance(item, str) and item == "":
            continue
        filtered.append(item)
    return filtered


def _normalize_insert_rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    rows = value
    if not isinstance(rows, list):
        rows = [rows]

    normalized_rows: list[list[Any]] = []
    for row in rows:
        current = row
        while isinstance(current, list) and len(current) == 1 and isinstance(current[0], list):
            current = current[0]
        if not isinstance(current, list):
            current = [current]
        normalized_row: list[Any] = []
        for cell in current:
            if cell is None:
                continue
            if isinstance(cell, str) and cell == "":
                continue
            normalized_row.append(normalize_value(_normalize_scalar(cell)))
        if normalized_row:
            normalized_rows.append(normalized_row)
    return normalized_rows


def _arguments_equal(expected_args: dict[str, Any], predicted_args: dict[str, Any]) -> bool:
    if expected_args == predicted_args:
        return True

    if _looks_like_bfcl_sql_arguments(expected_args) and _looks_like_bfcl_sql_arguments(
        predicted_args
    ):
        return _bfcl_sql_arguments_equal(expected_args, predicted_args)
    if _query_like_arguments_equal(expected_args, predicted_args):
        return True
    return False


def _query_like_arguments_equal(
    expected_args: dict[str, Any], predicted_args: dict[str, Any]
) -> bool:
    if set(expected_args.keys()) != set(predicted_args.keys()):
        return False
    if len(expected_args) != 1:
        return False
    key = next(iter(expected_args.keys()))
    if key.lower() not in {"query", "q", "keyword", "keywords", "search"}:
        return False
    exp = expected_args.get(key)
    pred = predicted_args.get(key)
    if not isinstance(exp, str) or not isinstance(pred, str):
        return False
    return _query_text_equivalent(exp, pred)


def _query_text_equivalent(expected: str, predicted: str) -> bool:
    exp_norm = _normalize_query_text(expected)
    pred_norm = _normalize_query_text(predicted)
    if not exp_norm or not pred_norm:
        return False
    if exp_norm == pred_norm:
        return True
    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True

    exp_tokens = set(exp_norm.split())
    pred_tokens = set(pred_norm.split())
    if not exp_tokens or not pred_tokens:
        return False
    inter = exp_tokens & pred_tokens
    union = exp_tokens | pred_tokens
    jaccard = len(inter) / len(union)
    return jaccard >= 0.6


def _normalize_query_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(
        r"[^\w\s]",
        " ",
        cleaned,
        flags=re.UNICODE,
    )
    tokens = [tok for tok in cleaned.split() if tok]
    stop = {
        "ara",
        "aramasi",
        "arama",
        "arastir",
        "araştır",
        "bul",
        "webde",
        "internette",
        "icin",
        "için",
        "the",
        "a",
        "an",
        "in",
        "on",
        "for",
    }
    tokens = [tok for tok in tokens if tok not in stop]
    return " ".join(tokens)


def _bfcl_sql_arguments_equal(
    expected_args: dict[str, Any], predicted_args: dict[str, Any]
) -> bool:
    expected = _strip_empty_optional_keys(expected_args)
    predicted = _strip_empty_optional_keys(predicted_args)

    expected_keyword = str(expected.get("sql_keyword", ""))
    predicted_keyword = str(predicted.get("sql_keyword", ""))
    if expected_keyword != predicted_keyword:
        return False
    if expected.get("table_name") != predicted.get("table_name"):
        return False

    for key in sorted(set(expected.keys()) | set(predicted.keys())):
        if key in {"sql_keyword", "table_name"}:
            continue
        exp_value = expected.get(key)
        pred_value = predicted.get(key)

        if key == "columns" and expected_keyword == "DELETE":
            if exp_value and not pred_value:
                continue
        if exp_value is None or pred_value is None:
            return exp_value == pred_value
        if key == "conditions":
            if isinstance(exp_value, list) and isinstance(pred_value, list):
                if sorted(exp_value) == sorted(pred_value):
                    continue
            if exp_value != pred_value:
                return False
            continue
        if key == "insert_values":
            if not _insert_values_equal(exp_value, pred_value):
                return False
            continue
        if exp_value != pred_value:
            return False
    return True


def _strip_empty_optional_keys(arguments: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"columns", "conditions", "insert_values", "update_values"}:
            if value is None or value == []:
                continue
        cleaned[key] = value
    return cleaned


def _insert_values_equal(expected: Any, predicted: Any) -> bool:
    if expected == predicted:
        return True
    if not isinstance(expected, list) or not isinstance(predicted, list):
        return False
    if len(predicted) == 1 and predicted[0] in expected:
        return True
    if len(expected) == 1 and expected[0] in predicted:
        return True
    return False


def _score_rank(
    result: tuple[bool, bool, bool, int, str | None]
) -> tuple[int, int, int, int]:
    exact_match, name_match, call_count_match, arg_match_count, _ = result
    return (
        1 if exact_match else 0,
        1 if name_match else 0,
        1 if call_count_match else 0,
        arg_match_count,
    )


def _normalize_sql_keyword(keyword: str) -> str:
    cleaned = " ".join(keyword.upper().split())
    if cleaned.startswith("INSERT"):
        return "INSERT"
    return cleaned


def _normalize_condition_scalar(raw: str) -> str:
    text = " ".join(raw.replace("\xa0", " ").strip().split())
    match = re.match(r"^(.*?)\s*(<=|>=|!=|=|<|>)\s*(.*)$", text)
    if not match:
        return text.lower()

    left_raw, operator, right_raw = match.groups()
    left = _normalize_identifier(left_raw)
    right = right_raw.strip()
    right = right.rstrip(";")
    if (
        len(right) >= 2
        and right[0] == right[-1]
        and right[0] in {"'", '"'}
    ):
        right = right[1:-1].strip()

    normalized_right = _normalize_scalar(right)
    if isinstance(normalized_right, str):
        normalized_right = normalized_right.replace("\xa0", " ").strip().lower()
    return f"{left}{operator}{normalized_right}"
