from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import ExplainableAgent
from .config import Settings
from .lmstudio_client import LMStudioClient
from .report import write_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aciklanabilir Ajan MVP (LM Studio yerel LLM)."
    )
    parser.add_argument("--task", type=str, help="Ajan icin gorev metni.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Kullanilacak model id (varsayilan: AGENT_MODEL veya gpt-oss-20b).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="gpt-oss-20b icin thinking seviyesi.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Ajanin maksimum adim sayisi.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LM Studio API taban adresi, ornek: http://localhost:1234/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LM Studio API anahtari (varsayilan: lm-studio).",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Dosya araclari icin calisma klasoru koku.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Calisma trace ve raporlarinin yazilacagi klasor.",
    )
    parser.add_argument(
        "--sqlite-db",
        type=str,
        default=None,
        help="Workspace'e gore SQLite DB yolu (varsayilan: data/agent.db).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="LM Studio'da yuklu modelleri listele.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.sqlite_db:
        os.environ["AGENT_SQLITE_DB"] = args.sqlite_db
    if args.base_url:
        settings = settings.with_overrides(base_url=args.base_url)
    if args.api_key:
        settings = settings.with_overrides(api_key=args.api_key)
    if args.model:
        settings = settings.with_overrides(requested_model=args.model)
    if args.reasoning_effort:
        settings = settings.with_overrides(reasoning_effort=args.reasoning_effort)
    if args.max_steps is not None:
        settings = settings.with_overrides(max_steps=args.max_steps)
    if args.workspace:
        settings = settings.with_overrides(workspace_root=Path(args.workspace).resolve())
    if args.runs_dir:
        settings = settings.with_overrides(runs_dir=Path(args.runs_dir).resolve())

    client = LMStudioClient(base_url=settings.base_url, api_key=settings.api_key)

    if args.list_models:
        try:
            model_ids = client.list_models()
        except Exception as exc:  # noqa: BLE001
            print(f"Model listesi alinamadi: {exc}")
            return 1
        if not model_ids:
            print("LM Studio'da yuklu model yok.")
            return 0
        print("Yuklu modeller:")
        for mid in model_ids:
            print(f"- {mid}")
        return 0

    if not args.task:
        parser.error("--task zorunludur (yalniz --list-models kullanilmiyorsa).")

    agent = ExplainableAgent(settings=settings, client=client)
    try:
        trace = agent.run(args.task)
    except Exception as exc:  # noqa: BLE001
        print(f"Ajan calismasi basarisiz: {exc}")
        return 1

    trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)
    print("Nihai cevap:")
    print(trace.final_answer)
    print("")
    print(f"Trace (kisa): {trace_path}")
    print(f"Rapor: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
