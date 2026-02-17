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
        "Adim 1: LM Studio Model Kontrolu",
    )
    if code != 0 or MODEL not in out:
        print(f"HATA: '{MODEL}' modeli LM Studio listesinde bulunamadi.")
        failed = True
        print("\nSONUC: BASARISIZ (LM Studio baglantisi veya model yuklemesi gerekli)")
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
        "Adim 2: Demo SQLite Baslatma",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("HATA: baslatma adiminda sqlite araci kullanilmadi.")
        failed = True
    else:
        print(f"BASARILI: sqlite araci bulundu ({trace})")

    code, _, _ = run_cmd(
        [
            *common,
            "--task",
            "sqlite_list_tables",
        ],
        "Adim 3: Tablo Listeleme",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("HATA: tablo listeleme adiminda sqlite araci kullanilmadi.")
        failed = True
    else:
        print(f"BASARILI: sqlite araci bulundu ({trace})")

    code, out, _ = run_cmd(
        [
            *common,
            "--task",
            "sqlite_query: SELECT name, city FROM customers ORDER BY id;",
        ],
        "Adim 4: Okuma Sorgusu",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("HATA: okuma sorgusu adiminda sqlite araci kullanilmadi.")
        failed = True
    else:
        print(f"BASARILI: sqlite araci bulundu ({trace})")
    if "Acme" not in out and "Istanbul" not in out:
        print("UYARI: cikti demo satirlari (Acme/Istanbul) icermiyor.")

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
        "Adim 5: Toplulastirma Sorgusu",
    )
    trace = latest_trace_path()
    if code != 0 or not trace or not trace_has_sqlite_tool(trace):
        print("HATA: toplulastirma adiminda sqlite araci kullanilmadi.")
        failed = True
    else:
        print(f"BASARILI: sqlite araci bulundu ({trace})")

    if failed:
        print("\nSONUC: BASARISIZ")
        return 1
    print("\nSONUC: BASARILI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
