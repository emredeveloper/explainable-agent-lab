"""Opt-in live checks; requires a running OpenAI-compatible model server.

Runs only local fixture tools. Saves full traces and a machine-readable summary.
These checks are separate from pytest because model outputs are nondeterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/live-checks"))
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    output = args.output.resolve() / uuid4().hex[:8]
    workspace = output / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "inventory.txt").write_text(
        "Product: amber bolts\nQuantity: 37\nUnit price: 12\n", encoding="utf-8"
    )
    settings = Settings.from_env().with_overrides(
        base_url=args.base_url,
        requested_model=args.model,
        api_key="local",
        workspace_root=workspace,
        runs_dir=output,
        temperature=0.0,
        max_steps=5,
        request_timeout=60,
        max_retries=0,
        chaos_mode=False,
        use_native_tools=False,
        stream=False,
    )
    rows = []
    cases = [
        ("math", "calculate_math: (215*4)-12", "848", False),
        (
            "file_math",
            "Read inventory.txt and calculate the total value (quantity multiplied by unit price).",
            "444",
            False,
        ),
        ("stream", "calculate_math: 37*12", "444", True),
        (
            "recovery",
            "Use flaky_lookup to retrieve the warehouse code. Retry if it fails.",
            "AMBER-7391",
            False,
        ),
        (
            "permanent_error",
            "Use broken_lookup to retrieve the warehouse code. Report failure if it is unavailable.",
            "",
            False,
        ),
    ]
    for repetition in range(1, args.repeat + 1):
        for name, task, expected, streaming in cases:
            registry = ToolRegistry.from_global()
            attempts = [0]

            @registry.define_tool(
                "flaky_lookup",
                "Returns a warehouse code; retry temporary failures.",
                "No input required.",
                requires_input=False,
            )
            def flaky_lookup(_text, _root, attempts=attempts):
                attempts[0] += 1
                if attempts[0] == 1:
                    raise TimeoutError("temporary lookup failure; retry")
                return "Warehouse code: AMBER-7391"

            @registry.define_tool(
                "broken_lookup",
                "Warehouse lookup that is currently unavailable.",
                "No input required.",
                requires_input=False,
            )
            def broken_lookup(_text, _root):
                raise ConnectionError("warehouse lookup unavailable")

            if name == "permanent_error":
                registry = ToolRegistry(
                    {"broken_lookup": registry.specs["broken_lookup"]}
                )

            print(f"CHECK {name} ({repetition}/{args.repeat})", flush=True)
            row = {"case": name, "repetition": repetition}
            try:
                trace = ExplainableAgent(
                    settings.with_overrides(stream=streaming), tool_registry=registry
                ).run(task)
                trace_path, _ = write_run_artifacts(trace, output)
                tool_names = [s.decision.tool_name for s in trace.steps]
                checks = {
                    "expected_answer": expected in trace.final_answer,
                    "no_raw_protocol": "<|tool_call" not in trace.final_answer,
                }
                if name == "file_math":
                    checks["used_both_tools"] = all(
                        t in tool_names for t in ("read_text_file", "calculate_math")
                    )
                    if "$" in trace.final_answer:
                        checks[
                            "unsupported_currency_flagged"
                        ] = not trace.faithfulness.likely_faithful
                if name == "stream":
                    checks["usage_reported"] = trace.total_usage["total_tokens"] > 0
                if name == "recovery":
                    checks["observed_recovery"] = (
                        trace.recovery_counts["successful_retries"] >= 1
                    )
                if name == "permanent_error":
                    checks["failed_tool_exercised"] = "broken_lookup" in tool_names
                    checks["no_false_recovery"] = (
                        trace.recovery_counts["successful_retries"] == 0
                    )
                    checks[
                        "no_false_faithfulness"
                    ] = not trace.faithfulness.likely_faithful
                row.update(
                    passed=all(checks.values()),
                    checks=checks,
                    answer=trace.final_answer,
                    usage=trace.total_usage,
                    recovery=trace.recovery_counts,
                    warnings=trace.errors,
                    trace=str(trace_path),
                )
            except Exception as exc:
                row.update(passed=False, error=repr(exc))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            (output / "summary.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )

        # Both planning and execution use the real model. Only the first agent
        # can read the fixture; the second must receive its result via context.
        print("CHECK dependent_team", flush=True)
        reader_registry = ToolRegistry()

        @reader_registry.define_tool(
            "get_secret_code",
            "Read the secret warehouse code.",
            "No input required.",
            requires_input=False,
        )
        def get_secret_code(_text, _root):
            return "Warehouse code: AMBER-7391"

        client = OpenAICompatClient(args.base_url, "local", timeout=60, max_retries=0)
        reader = ExplainableAgent(settings, tool_registry=reader_registry)
        reporter = ExplainableAgent(settings, tool_registry=ToolRegistry())
        orchestrator = TeamOrchestrator(
            client,
            {
                "reader": (
                    "Can retrieve the secret warehouse code using get_secret_code.",
                    reader,
                ),
                "reporter": (
                    "Has no tools. Uses the reader's previous result to report the exact warehouse code.",
                    reporter,
                ),
            },
        )
        row = {"case": "dependent_team", "repetition": repetition}
        try:
            trace = orchestrator.run(
                "First ask reader to retrieve the warehouse code. Then ask reporter to report that exact code using reader's result. Use both agents in that order.",
                args.model,
            )
            trace_path, _ = write_orchestrator_artifacts(trace, output)
            reporter_results = [
                st.trace.final_answer
                for st in trace.subtasks
                if st.agent_name == "reporter"
            ]
            passed = bool(reporter_results) and all(
                "AMBER-7391" in answer for answer in reporter_results
            )
            row.update(
                passed=passed,
                answer=trace.final_synthesis,
                trace=str(trace_path),
                reporter_results=reporter_results,
            )
        except Exception as exc:
            row.update(passed=False, error=repr(exc))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
        (output / "summary.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    print(f"Results: {output / 'summary.json'}", flush=True)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
