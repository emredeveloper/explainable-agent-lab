"""Compare local model families against exact, fixture-backed answer contracts.

No model judges its own answers. Checks cover only the declared fixture facts.
Run from a repository checkout; models must already be installed in Ollama.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainable_agent import (
    ExplainableAgent,
    Settings,
    ToolRegistry,
    write_run_artifacts,
)
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.orchestrator import TeamOrchestrator
from explainable_agent.report import write_orchestrator_artifacts


def exact_value(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(exact_value(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and actual == expected
        )
    return type(actual) is type(expected) and actual == expected


def grade_answer(answer, expected):
    try:
        actual = json.loads(answer)
    except (ValueError, TypeError):
        return {"valid_json": False, "exact_facts": False}
    return {"valid_json": True, "exact_facts": exact_value(actual, expected)}


def registry_for(case, code):
    builtins = ToolRegistry.from_global()
    registry = ToolRegistry(
        {
            name: builtins.specs[name]
            for name in ("read_text_file", "calculate_math", "list_workspace_files")
        }
    )
    if case in {"recovery", "unavailable", "team"}:
        registry = ToolRegistry()
    attempts = [0]

    @registry.define_tool(
        "warehouse_lookup",
        "Retrieve the warehouse code. Retry temporary errors.",
        "No input required.",
        requires_input=False,
    )
    def lookup(_text, _root):
        attempts[0] += 1
        if case == "unavailable":
            raise ConnectionError("warehouse unavailable; no code was retrieved")
        if case == "recovery" and attempts[0] == 1:
            raise TimeoutError("temporary error; retry warehouse_lookup")
        return json.dumps({"code": code})

    if case not in {"recovery", "unavailable", "team"}:
        registry.specs.pop("warehouse_lookup")
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--output", type=Path, default=Path("runs/model-matrix"))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    output = args.output.resolve() / uuid4().hex[:8]
    output.mkdir(parents=True)
    with urllib.request.urlopen(
        args.base_url.removesuffix("/v1") + "/api/tags", timeout=10
    ) as response:
        metadata = json.load(response)
    installed = {m["name"]: m for m in metadata["models"]}
    missing = set(args.models) - installed.keys()
    if missing:
        parser.error(f"Models are not installed: {sorted(missing)}")
    rows = []
    contract = " Return only the requested JSON object in your final answer, with no Markdown, explanation, or additional keys. Use null for missing facts; never infer a currency."
    cases = [
        (
            "math",
            'Calculate (215*4)-12 using a tool. Final object: {"result": number}.',
            {"result": 848},
            False,
        ),
        (
            "file_total",
            'Read inventory.txt and calculate quantity times unit price. Final object: {"total": number, "currency": string or null}.',
            {"total": 444, "currency": None},
            False,
        ),
        (
            "turkish",
            'inventory.txt dosyasını oku. Stok adedini ve birim fiyatını yaz. Son nesne: {"adet": sayı, "birim_fiyat": sayı}.',
            {"adet": 37, "birim_fiyat": 12},
            False,
        ),
        (
            "missing_fact",
            'Read inventory.txt and report the supplier email. Final object: {"supplier_email": string or null}.',
            {"supplier_email": None},
            False,
        ),
        (
            "conflict",
            'Read old.txt and current.txt. Use the record with the later date for the quantity. Final object: {"quantity": number, "source": filename}.',
            {"quantity": 19, "source": "current.txt"},
            False,
        ),
        (
            "stream",
            'Calculate 37*12 using a tool. Final object: {"result": number}.',
            {"result": 444},
            True,
        ),
        (
            "recovery",
            'Use warehouse_lookup to retrieve the code, retry temporary errors. Final object: {"code": string}.',
            None,
            False,
        ),
        (
            "unavailable",
            'Use warehouse_lookup. If it is unavailable, report that status without inventing a code. Final object: {"status": "unavailable", "code": null}.',
            {"status": "unavailable", "code": None},
            False,
        ),
    ]
    for model in args.models:
        for repetition in range(1, args.repeat + 1):
            code = "WH-" + uuid4().hex[:8].upper()
            workspace = output / f"workspace-{args.models.index(model)}-{repetition}"
            workspace.mkdir()
            for name, content in {
                "inventory.txt": "Product: amber bolts\nQuantity: 37\nUnit price: 12\n",
                "old.txt": "Date: 2026-09-01\nQuantity: 41\n",
                "current.txt": "Date: 2026-09-07\nQuantity: 19\n",
            }.items():
                (workspace / name).write_text(content, encoding="utf-8")
            settings = Settings.from_env().with_overrides(
                base_url=args.base_url,
                api_key="local",
                requested_model=model,
                workspace_root=workspace,
                runs_dir=output,
                max_steps=5,
                temperature=0.0,
                chaos_mode=False,
                stream=False,
                use_native_tools=False,
                request_timeout=120,
                max_retries=0,
            )
            for name, task, expected, streaming in cases:
                expected = {"code": code} if name == "recovery" else expected
                row = {
                    "model": model,
                    "case": name,
                    "repetition": repetition,
                    "expected": expected,
                }
                print(f"CHECK {model} {name} {repetition}/{args.repeat}", flush=True)
                started = time.perf_counter()
                try:
                    trace = ExplainableAgent(
                        settings.with_overrides(stream=streaming),
                        tool_registry=registry_for(name, code),
                    ).run(task + contract)
                    path, _ = write_run_artifacts(trace, output)
                    checks = grade_answer(trace.final_answer, expected)
                    tools = [s.decision.tool_name for s in trace.steps]
                    checks["required_tool"] = (
                        "warehouse_lookup"
                        if name in {"recovery", "unavailable"}
                        else "calculate_math"
                        if name in {"math", "stream", "file_total"}
                        else "read_text_file"
                    ) in tools
                    if name == "stream":
                        checks["usage_reported"] = trace.total_usage["total_tokens"] > 0
                    if name == "recovery":
                        checks["recovered"] = (
                            trace.recovery_counts["successful_retries"] == 1
                        )
                    if name == "unavailable":
                        checks["no_false_recovery"] = (
                            trace.recovery_counts["successful_retries"] == 0
                        )
                    row.update(
                        passed=all(checks.values()),
                        checks=checks,
                        answer=trace.final_answer,
                        trace=str(path),
                        tokens=trace.total_usage,
                        warnings=trace.errors,
                    )
                except Exception as exc:
                    row.update(passed=False, error=repr(exc))
                row["seconds"] = round(time.perf_counter() - started, 2)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                save(output, rows, installed, args)
            row = {
                "model": model,
                "case": "team",
                "repetition": repetition,
                "expected": {"code": code},
            }
            started = time.perf_counter()
            print(f"CHECK {model} team {repetition}/{args.repeat}", flush=True)
            try:
                team = TeamOrchestrator(
                    OpenAICompatClient(
                        args.base_url, "local", timeout=120, max_retries=0
                    ),
                    {
                        "reader": (
                            "Retrieves code with warehouse_lookup.",
                            ExplainableAgent(
                                settings, tool_registry=registry_for("team", code)
                            ),
                        ),
                        "reporter": (
                            "No tools; reports reader's result.",
                            ExplainableAgent(settings, tool_registry=ToolRegistry()),
                        ),
                    },
                )
                trace = team.run(
                    'First reader retrieves the warehouse code. Then reporter reports that code using reader\'s result. Assign exactly those two subtasks in order. Both agents and the final synthesis must answer with {"code": string} only.'
                    + contract,
                    model,
                )
                path, _ = write_orchestrator_artifacts(trace, output)
                checks = grade_answer(trace.final_synthesis, {"code": code})
                checks["both_agents_in_order"] = [
                    s.agent_name for s in trace.subtasks
                ] == ["reader", "reporter"]
                checks["reporter_correct"] = bool(trace.subtasks) and all(
                    grade_answer(s.trace.final_answer, {"code": code})["exact_facts"]
                    for s in trace.subtasks
                    if s.agent_name == "reporter"
                )
                row.update(
                    passed=all(checks.values()),
                    checks=checks,
                    answer=trace.final_synthesis,
                    trace=str(path),
                )
            except Exception as exc:
                row.update(passed=False, error=repr(exc))
            row["seconds"] = round(time.perf_counter() - started, 2)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            save(output, rows, installed, args)
    print(f"Results: {output / 'summary.json'}", flush=True)
    return 0 if all(row["passed"] for row in rows) else 1


def save(output, rows, installed, args):
    (output / "summary.json").write_text(
        json.dumps(
            {
                "grading": "exact-json-fixture-v1",
                "models": {m: installed[m] for m in args.models},
                "temperature": 0.0,
                "repeat": args.repeat,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
