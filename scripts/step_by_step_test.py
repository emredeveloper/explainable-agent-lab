from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MODEL = "gpt-oss-20b"


def run_cmd(args: list[str], step_name: str) -> tuple[int, str, str]:
    print(f"\n=== {step_name} ===")
    print(">", " ".join(args))
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    print(f"exit_code={proc.returncode}")
    return proc.returncode, proc.stdout, proc.stderr


def latest_trace_path() -> Path | None:
    run_root = ROOT / "runs"
    if not run_root.exists():
        return None
    traces = sorted(run_root.glob("*/trace.json"))
    return traces[-1] if traces else None


def check_trace_has_tool_call(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    actions: list[str] = []
    for step in data["steps"]:
        action = step.get("action")
        if action is None and isinstance(step.get("decision"), dict):
            action = step["decision"].get("action")
        if isinstance(action, str):
            actions.append(action)
    return "tool_call" in actions and "final_answer" in actions


def main() -> int:
    failed = False

    code, _, _ = run_cmd(
        [
            PYTHON,
            "-m",
            "py_compile",
            "explainable_agent/__init__.py",
            "explainable_agent/config.py",
            "explainable_agent/cli.py",
            "explainable_agent/agent.py",
            "explainable_agent/tools.py",
            "explainable_agent/openai_client.py",
        ],
        "Step 1: Syntax Check",
    )
    if code != 0:
        failed = True

    code, out, _ = run_cmd(
        [PYTHON, "-m", "explainable_agent.cli", "--list-models"],
        "Step 2: Model Check",
    )
    if code != 0 or MODEL not in out:
        print(f"FAILED: model '{MODEL}' was not found in the server list.")
        failed = True
        print("\nRESULT: FAILED (API connection or model loading required)")
        return 1

    code, out, _ = run_cmd(
        [
            PYTHON,
            "-m",
            "explainable_agent.cli",
            "--model",
            MODEL,
            "--reasoning-effort",
            "high",
            "--max-steps",
            "4",
            "--task",
            "calculate_math: (215*4)-12",
        ],
        "Step 3: Math Tool Task",
    )
    if code != 0 or "848" not in out:
        print("FAILED: expected '848' missing from the math task output.")
        failed = True
    trace = latest_trace_path()
    if not trace:
        print("FAILED: trace.json was not produced after the math task.")
        failed = True
    elif not check_trace_has_tool_call(trace):
        print(f"FAILED: expected tool_call + final_answer inside {trace}.")
        failed = True
    else:
        print(f"OK: trace actions are correct ({trace})")

    code, _, _ = run_cmd(
        [
            PYTHON,
            "-m",
            "explainable_agent.cli",
            "--model",
            MODEL,
            "--reasoning-effort",
            "high",
            "--max-steps",
            "4",
            "--task",
            "list_workspace_files: .|*.py",
        ],
        "Step 4: File Listing Tool Task",
    )
    if code != 0:
        failed = True
    trace = latest_trace_path()
    if not trace:
        print("FAILED: trace.json was not produced after the listing task.")
        failed = True
    elif not check_trace_has_tool_call(trace):
        print(f"FAILED: expected tool_call + final_answer inside {trace}.")
        failed = True
    else:
        print(f"OK: trace actions are correct ({trace})")

    if failed:
        print("\nRESULT: FAILED")
        return 1
    print("\nRESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
