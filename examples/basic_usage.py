from __future__ import annotations

import argparse
from pathlib import Path

from explainable_agent.agent import ExplainableAgent
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.report import write_run_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small ExplainableAgent demo.")
    parser.add_argument(
        "--task",
        default="calculate_math: (215*4)-12",
        help="Task to run. Defaults to a deterministic built-in tool call.",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="API key for the endpoint.")
    parser.add_argument("--model", default=None, help="Model name.")
    parser.add_argument("--workspace", default=None, help="Workspace for file tools.")
    parser.add_argument("--runs-dir", default=None, help="Directory for run artifacts.")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum agent steps.")
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Use JSON decision mode instead of native function calling.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print rich step panels."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env().with_overrides(
        base_url=args.base_url,
        api_key=args.api_key,
        requested_model=args.model,
        max_steps=args.max_steps,
        workspace_root=Path(args.workspace).resolve() if args.workspace else None,
        runs_dir=Path(args.runs_dir).resolve() if args.runs_dir else None,
        use_native_tools=not args.json_mode,
    )
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)
    agent = ExplainableAgent(settings=settings, client=client, verbose=args.verbose)

    trace = agent.run(args.task)
    trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)

    print("Final answer:")
    print(trace.final_answer)
    print(f"Trace: {trace_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
