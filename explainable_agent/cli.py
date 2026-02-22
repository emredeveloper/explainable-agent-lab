from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import ExplainableAgent
from .config import Settings
from .openai_client import OpenAICompatClient
from .report import write_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explainable Agent MVP (OpenAI-compatible local LLM)."
    )
    parser.add_argument("--task", type=str, help="Task text for the agent.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to use (default: AGENT_MODEL or gpt-oss-20b).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort level.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of agent steps.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL, e.g. http://localhost:1234/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: local).",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root directory for file tools.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Directory for writing run traces and reports.",
    )
    parser.add_argument(
        "--sqlite-db",
        type=str,
        default=None,
        help="SQLite DB path relative to workspace (default: data/agent.db).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List installed models on the server.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print steps to terminal with colors.",
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="Enable Chaos Mode to simulate random tool errors for testing self-healing.",
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
    if args.chaos:
        settings = settings.with_overrides(chaos_mode=True)

    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)

    if args.list_models:
        try:
            model_ids = client.list_models()
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to get model list: {exc}")
            return 1
        if not model_ids:
            print("No models installed on the server.")
            return 0
        print("Installed models:")
        for mid in model_ids:
            print(f"- {mid}")
        return 0

    if not args.task:
        parser.error("--task is required (unless using --list-models).")

    agent = ExplainableAgent(settings=settings, client=client, verbose=args.verbose)
    try:
        trace = agent.run(args.task)
    except Exception as exc:  # noqa: BLE001
        print(f"Agent execution failed: {exc}")
        return 1

    trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)
    print("Final answer:")
    print(trace.final_answer)
    print("")
    print(f"Trace (compact): {trace_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
