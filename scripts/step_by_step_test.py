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

    code, _, _ = run_cmd([PYTHON, "-m", "pytest", "-q"], "Adim 1: Birim Testleri")
    if code != 0:
        failed = True

    code, out, _ = run_cmd(
        [PYTHON, "-m", "explainable_agent.cli", "--list-models"],
        "Adim 2: Model Kontrolu",
    )
    if code != 0 or MODEL not in out:
        print(f"HATA: '{MODEL}' modeli sunucu listesinde bulunamadi.")
        failed = True
        print("\nSONUC: BASARISIZ (API baglantisi veya model yuklemesi gerekli)")
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
        "Adim 3: Matematik Arac Gorevi",
    )
    if code != 0 or "848" not in out:
        print("HATA: matematik gorevi cikti metninde beklenen '848' yok.")
        failed = True
    trace = latest_trace_path()
    if not trace:
        print("HATA: matematik gorevinden sonra trace.json uretilemedi.")
        failed = True
    elif not check_trace_has_tool_call(trace):
        print(f"HATA: {trace} icinde tool_call + final_answer bekleniyordu.")
        failed = True
    else:
        print(f"BASARILI: trace aksiyonlari dogru ({trace})")

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
        "Adim 4: Dosya Listeleme Arac Gorevi",
    )
    if code != 0:
        failed = True
    trace = latest_trace_path()
    if not trace:
        print("HATA: listeleme gorevinden sonra trace.json uretilemedi.")
        failed = True
    elif not check_trace_has_tool_call(trace):
        print(f"HATA: {trace} icinde tool_call + final_answer bekleniyordu.")
        failed = True
    else:
        print(f"BASARILI: trace aksiyonlari dogru ({trace})")

    if failed:
        print("\nSONUC: BASARISIZ")
        return 1
    print("\nSONUC: BASARILI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
