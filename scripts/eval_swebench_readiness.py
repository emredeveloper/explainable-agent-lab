from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainable_agent.config import Settings
from explainable_agent.dataset_adapters import (
    load_dataset_with_adapter,
    resolve_dataset_format,
)
from explainable_agent.json_utils import parse_json_object_relaxed


RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "swebench_readiness_plan",
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "properties": {
                        "root_cause_hypothesis": {"type": "string"},
                        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["root_cause_hypothesis", "risk"],
                    "additionalProperties": False,
                },
                "files_to_inspect": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "first_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["analysis", "files_to_inspect", "first_actions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite readiness evaluator (JSON-plan quality)."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["auto", "swebench_lite"],
        default="auto",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--runs-dir", type=str, default="runs/evals")
    parser.add_argument(
        "--language", 
        type=str, 
        default="en", 
        choices=["en"], 
        help="Output language (en)."
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help=(
            "Maximum token limit. If not provided/0, max_tokens is not sent; "
            "model stops when response is complete."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _build_messages(problem_statement: str, language: str = "en") -> list[dict[str, str]]:
    sys_prompt = (
        "You are a software issue triage assistant. Return only valid JSON.\n"
        'Format: {"analysis":{"root_cause_hypothesis":"...","risk":"low|medium|high"},'
        '"files_to_inspect":["..."],"first_actions":["..."]}'
    )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": problem_statement},
    ]


def _extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    return parse_json_object_relaxed(text)


def _score_plan(payload: dict[str, Any] | None) -> tuple[bool, bool, bool, str | None]:
    if payload is None:
        return False, False, False, "parse_error"
    analysis = payload.get("analysis")
    files = payload.get("files_to_inspect")
    actions = payload.get("first_actions")

    has_analysis = isinstance(analysis, dict) and bool(
        str(analysis.get("root_cause_hypothesis", "")).strip()
    )
    has_files = isinstance(files, list) and len([f for f in files if str(f).strip()]) >= 1
    has_actions = isinstance(actions, list) and len(
        [a for a in actions if str(a).strip()]
    ) >= 1
    if has_analysis and has_files and has_actions:
        return True, True, True, None
    if not has_analysis:
        return False, has_files, has_actions, "missing_analysis"
    if not has_files:
        return has_analysis, False, has_actions, "missing_files"
    return has_analysis, has_files, False, "missing_actions"


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    dataset_path = Path(args.dataset)
    dataset_format = resolve_dataset_format(dataset_path, args.dataset_format)
    if dataset_format != "swebench_lite":
        print("This script only supports the swebench_lite format.")
        return 1

    output = load_dataset_with_adapter(
        dataset_path=dataset_path,
        dataset_format=dataset_format,
        limit=args.limit,
    )
    rows = output.rows
    if not rows:
        print("No SWE-bench data rows found.")
        return 1

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    model = args.model or settings.requested_model

    details: list[dict[str, Any]] = []
    err = Counter()
    valid_json = 0
    complete_plan = 0
    for idx, row in enumerate(rows):
        request: dict[str, Any] = {
            "model": model,
            "messages": _build_messages(str(row["query"]), language=args.language),
            "temperature": 0,
        }
        if args.max_completion_tokens is not None and args.max_completion_tokens > 0:
            request["max_tokens"] = args.max_completion_tokens
        try:
            response = client.chat.completions.create(
                **request,
                response_format=RESPONSE_SCHEMA,
            )
        except Exception:
            response = client.chat.completions.create(**request)
        raw = response.choices[0].message.content or ""
        payload, parse_method = _extract_json(raw)
        has_analysis, has_files, has_actions, error = _score_plan(payload)
        if payload is not None:
            valid_json += 1
        if has_analysis and has_files and has_actions:
            complete_plan += 1
        if error:
            err[error] += 1
        details.append(
            {
                "index": idx,
                "source_row_id": row.get("source_row_id"),
                "query": row.get("query"),
                "raw_model_output": raw,
                "parsed_json": payload,
                "parse_method": parse_method,
                "has_analysis": has_analysis,
                "has_files_to_inspect": has_files,
                "has_first_actions": has_actions,
                "error_type": error,
            }
        )

    total = len(rows)
    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_format": dataset_format,
        "task_type": "swebench_patch",
        "model": model,
        "sample_count": total,
        "valid_json_rate": round(valid_json / total, 4),
        "complete_plan_rate": round(complete_plan / total, 4),
        "error_breakdown": dict(err),
    }

    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = runs_root / f"swebench_readiness_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Evaluation completed.")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Details: {out_dir / 'details.json'}")
    print(
        f"Scores -> valid_json: {summary['valid_json_rate']}, "
        f"complete_plan: {summary['complete_plan_rate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
