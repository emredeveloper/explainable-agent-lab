from __future__ import annotations

import argparse
import warnings

from explainable_agent.agent import ExplainableAgent
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.orchestrator import TeamOrchestrator
from explainable_agent.report import write_orchestrator_artifacts

warnings.filterwarnings("ignore", category=ResourceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-agent demo.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="API key for the endpoint.")
    parser.add_argument("--model", default=None, help="Model name.")
    parser.add_argument(
        "--max-steps", type=int, default=3, help="Sub-agent step limit."
    )
    parser.add_argument(
        "--task",
        default=(
            "Use the researcher agent to read README.md and summarize the project "
            "positioning. Use the db_expert agent to run sqlite_init_demo and then "
            "sqlite_query: select name, city from customers order by id. Combine "
            "both results in the final synthesis."
        ),
        help="Main orchestration task.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print orchestration panels."
    )
    return parser.parse_args()


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

    research_agent = ExplainableAgent(settings=settings, client=client, verbose=False)
    database_agent = ExplainableAgent(settings=settings, client=client, verbose=False)
    agents_team = {
        "researcher": (
            "Can read repository files and summarize project context.",
            research_agent,
        ),
        "db_expert": (
            "Can initialize and query the bundled SQLite demo database.",
            database_agent,
        ),
    }

    orchestrator = TeamOrchestrator(
        client=client, agents=agents_team, verbose=args.verbose
    )
    trace = orchestrator.run(
        main_task=args.task,
        requested_model=settings.requested_model,
    )
    trace_path, report_path = write_orchestrator_artifacts(trace, settings.runs_dir)

    print("Final synthesis:")
    print(trace.final_synthesis)
    print(f"Trace: {trace_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
