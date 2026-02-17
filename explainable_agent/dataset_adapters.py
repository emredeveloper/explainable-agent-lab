from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .eval_tool_calls import load_bfcl_sql_samples, load_eval_samples


TaskType = Literal["tool_call", "swebench_patch"]


@dataclass
class AdapterOutput:
    dataset_format: str
    task_type: TaskType
    rows: list[dict[str, Any]]


def resolve_dataset_format(dataset_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    if dataset_path.name == "BFCL_v3_sql.json":
        sibling_answer = dataset_path.parent / "possible_answer" / "BFCL_v3_sql.json"
        if sibling_answer.exists():
            return "bfcl_sql"
    if "swe-bench" in dataset_path.name.lower() or "swebench" in dataset_path.name.lower():
        return "swebench_lite"
    return "jsonl"


def load_dataset_with_adapter(
    *,
    dataset_path: Path,
    dataset_format: str,
    answers_path: Path | None = None,
    limit: int | None = None,
) -> AdapterOutput:
    if dataset_format == "bfcl_sql":
        if answers_path is None:
            answers_path = dataset_path.parent / "possible_answer" / "BFCL_v3_sql.json"
        rows = load_bfcl_sql_samples(
            question_path=dataset_path,
            answer_path=answers_path,
            limit=limit,
        )
        return AdapterOutput(dataset_format=dataset_format, task_type="tool_call", rows=rows)

    if dataset_format == "swebench_lite":
        rows = load_swebench_lite_samples(dataset_path=dataset_path, limit=limit)
        return AdapterOutput(
            dataset_format=dataset_format,
            task_type="swebench_patch",
            rows=rows,
        )

    rows = load_eval_samples(dataset_path, limit=limit)
    return AdapterOutput(dataset_format=dataset_format, task_type="tool_call", rows=rows)


def load_swebench_lite_samples(dataset_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    # Supports JSONL exports and JSON list exports of SWE-bench Lite-like rows.
    rows = _read_json_or_jsonl(dataset_path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
        problem_statement = str(row.get("problem_statement") or "").strip()
        repo = str(row.get("repo") or "").strip()
        base_commit = str(row.get("base_commit") or "").strip()
        if not problem_statement:
            continue
        normalized.append(
            {
                "source_dataset": "SWE-bench/SWE-bench_Lite",
                "source_row_id": instance_id or None,
                "query": problem_statement,
                "repo": repo,
                "base_commit": base_commit,
                "task_type": "swebench_patch",
                "raw_row": row,
            }
        )
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


def _read_json_or_jsonl(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else []
    rows: list[Any] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
