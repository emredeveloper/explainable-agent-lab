from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from explainable_agent.agent import ExplainableAgent
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.report import write_run_artifacts
from explainable_agent.tools import define_tool


@define_tool(
    name="get_user_email",
    description="Returns a mocked email address for a user ID.",
    usage_hint="Input should be just the numeric user ID, e.g. 101.",
)
def get_user_email(user_id_text: str, _: Path) -> str:
    user_db = {"101": "alice@example.com", "202": "bob@example.com"}
    user_id = user_id_text.strip()
    email = user_db.get(user_id)
    if email:
        return f"The email for user ID {user_id} is {email}."
    return f"ERROR: User ID '{user_id}' not found in the demo database."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact feature showcase.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="API key for the endpoint.")
    parser.add_argument("--model", default=None, help="Model name.")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum agent steps.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print rich step panels."
    )
    parser.add_argument(
        "--include-sqlite",
        action="store_true",
        help="Also run the SQLite tool scenario.",
    )
    parser.add_argument(
        "--include-custom",
        action="store_true",
        help="Also run the custom tool scenario.",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Also run the JSONL evaluation sample with limit=1.",
    )
    parser.add_argument(
        "--include-chaos",
        action="store_true",
        help="Also run one chaos-mode scenario.",
    )
    return parser.parse_args()


def run_agent_scenario(
    title: str,
    task: str,
    settings: Settings,
    client: OpenAICompatClient,
    verbose: bool,
) -> None:
    print("\n" + "=" * 60)
    print(f"SCENARIO: {title}")
    print("=" * 60)
    print(f"Task: {task}\n")

    agent = ExplainableAgent(settings=settings, client=client, verbose=verbose)
    trace = agent.run(task)
    trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)
    print(f"Final answer: {trace.final_answer}")
    print(f"Trace: {trace_path}")
    print(f"Report: {report_path}")


def run_evaluation_scenario(settings: Settings) -> None:
    eval_script = Path("scripts/eval_hf_tool_calls.py").resolve()
    dataset_path = Path("examples/custom_eval_sample.jsonl").resolve()
    cmd = [
        sys.executable,
        str(eval_script),
        "--dataset",
        str(dataset_path),
        "--base-url",
        settings.base_url,
        "--model",
        settings.requested_model,
        "--limit",
        "1",
        "--sampling",
        "head",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print("\n" + "=" * 60)
    print("SCENARIO: JSONL evaluation sample")
    print("=" * 60)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)


def main() -> int:
    args = parse_args()
    settings = Settings.from_env().with_overrides(
        base_url=args.base_url,
        api_key=args.api_key,
        requested_model=args.model,
        max_steps=args.max_steps,
        use_native_tools=True,
    )
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)

    scenarios = [
        ("Built-in math tool", "calculate_math: (512 * 4) + 128", settings, client)
    ]

    if args.include_sqlite:
        scenarios.append(
            (
                "SQLite demo tools",
                (
                    "sqlite_init_demo, then sqlite_query: "
                    "select name, city from customers order by id"
                ),
                settings,
                client,
            )
        )

    if args.include_custom:
        scenarios.append(("Custom tool", "get_user_email: 101", settings, client))

    if args.include_chaos:
        scenarios.append(
            (
                "Chaos mode",
                "calculate_math: (512 * 4) + 128",
                settings.with_overrides(chaos_mode=True),
                OpenAICompatClient(
                    base_url=settings.base_url, api_key=settings.api_key
                ),
            )
        )

    for title, task, scenario_settings, scenario_client in scenarios:
        run_agent_scenario(
            title, task, scenario_settings, scenario_client, verbose=args.verbose
        )

    if args.include_eval:
        run_evaluation_scenario(settings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
