from __future__ import annotations

import argparse
from pathlib import Path

from explainable_agent.agent import ExplainableAgent
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.report import write_run_artifacts
from explainable_agent.tools import define_tool


@define_tool(
    name="get_weather_info",
    description="Returns mocked weather information for a city.",
    usage_hint="Input is just the city name, e.g. Istanbul or Tokyo.",
)
def get_weather_info(city: str, _: Path) -> str:
    weather_db = {
        "istanbul": "Clear, 25 C",
        "london": "Rainy, 14 C",
        "new york": "Cloudy, 16 C",
        "tokyo": "Sunny, 22 C",
    }
    city_name = city.strip()
    result = weather_db.get(city_name.lower())
    if result:
        return f"The weather in {city_name} is currently {result}."
    return f"ERROR: Weather information not found for city '{city_name}'."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a custom tool example.")
    parser.add_argument("--city", default="Istanbul", help="City for the custom tool.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="API key for the endpoint.")
    parser.add_argument("--model", default=None, help="Model name.")
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
        use_native_tools=not args.json_mode,
    )
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)
    agent = ExplainableAgent(settings=settings, client=client, verbose=args.verbose)

    trace = agent.run(f"get_weather_info: {args.city}")
    trace_path, report_path = write_run_artifacts(trace, settings.runs_dir)

    print("Final answer:")
    print(trace.final_answer)
    print(f"Trace: {trace_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
