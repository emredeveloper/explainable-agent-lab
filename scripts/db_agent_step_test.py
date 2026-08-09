from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MODEL = "gpt-oss-20b"
DB_PATH = "data/demo_agent.db"


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


def trace_has_sqlite_tool(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    for step in data.get("steps", []):
        tool_name = step.get("tool")
        if isinstance(tool_name, str) and tool_name.startswith("sqlite_"):
            return True
        decision = step.get("decision")
        if isinstance(decision, dict):
            old_tool = decision.get("tool_name")
            if isinstance(old_tool, str) and old_tool.startswith("sqlite_"):
                return True
    return False


def main() -> int:
    failed = False

    code, out, _ = run_cmd(
        [PYTHON, "-m", "explainable_agent.cli", "--list-models"],
        "Step 1: Model Check",
    )
    if code != 0 or MODEL not in out:
        print(f"FAILED: model '{MODEL}' was not found in the server list.")
        failed = True
        print("\nRESULT: FAILED (API connection or model loading required)")
        return 1

    common = [
        PYTHON,
        "-m",
        "explainable_agent.cli",
        "--model",
        MODEL,
        "--reasoning-effort",
        "high",
        "--max-steps",
        "5",
        "--sqlite-db",
        DB_PATH,
    ]

    code, _, _ = run_cmd(
        [
            *common,
            "--task",
            "sqlite_init_demo",
        ],
        "Step 2: Demo SQLite Initialization",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("FAILED: no sqlite tool was used in the initialization step.")
        failed = True
    else:
        print(f"OK: sqlite tool found ({trace})")

    code, _, _ = run_cmd(
        [
            *common,
            "--task",
            "sqlite_list_tables",
        ],
        "Step 3: Table Listing",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("FAILED: no sqlite tool was used in the table listing step.")
        failed = True
    else:
        print(f"OK: sqlite tool found ({trace})")

    code, out, _ = run_cmd(
        [
            *common,
            "--task",
            "sqlite_query: SELECT name, city FROM customers ORDER BY id;",
        ],
        "Step 4: Read Query",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("FAILED: no sqlite tool was used in the read query step.")
        failed = True
    else:
        print(f"OK: sqlite tool found ({trace})")
    if "Acme" not in out and "Istanbul" not in out:
        print("WARNING: output does not contain the demo rows (Acme/Istanbul).")

    code, _, _ = run_cmd(
        [
            *common,
            "--task",
            (
                "sqlite_query: "
                "SELECT status, COUNT(*) as adet, ROUND(SUM(amount),2) as toplam "
                "FROM orders GROUP BY status ORDER BY status;"
            ),
        ],
        "Step 5: Aggregation Query",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("FAILED: no sqlite tool was used in the aggregation step.")
        failed = True
    else:
        print(f"OK: sqlite tool found ({trace})")

    if failed:
        print("\nRESULT: FAILED")
        return 1
    print("\nRESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
