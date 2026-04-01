from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> tuple[int, str, str]:
    print(f"$ {' '.join(cmd)}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        capture_output=capture,
        text=True,
    )
    if capture:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
    return proc.returncode, (proc.stdout or ""), (proc.stderr or "")


def main() -> int:
    tmp_root = ROOT / ".tmp"
    temp_dir = tmp_root / "system_tmp"
    for path in (tmp_root, temp_dir):
        path.mkdir(parents=True, exist_ok=True)

    test_env = {
        "TMP": str(temp_dir),
        "TEMP": str(temp_dir),
        "TMPDIR": str(temp_dir),
        "PYTHONIOENCODING": "utf-8",
    }

    checks: list[tuple[str, list[str]]] = [
        (
            "Compile",
            [
                sys.executable,
                "-m",
                "py_compile",
                "explainable_agent/__init__.py",
                "explainable_agent/config.py",
                "explainable_agent/cli.py",
                "explainable_agent/agent.py",
                "explainable_agent/tools.py",
                "explainable_agent/openai_client.py",
                "explainable_agent/eval_tool_calls.py",
                "explainable_agent/dataset_adapters.py",
                "explainable_agent/json_utils.py",
                "scripts/eval_hf_tool_calls.py",
                "scripts/eval_swebench_readiness.py",
            ],
        ),
    ]

    failed: list[str] = []
    for name, cmd in checks:
        print(f"\n== {name} ==")
        code, _out, _err = run(cmd, env=test_env, capture=False)
        if code != 0:
            failed.append(name)

    print("\n== Summary ==")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
