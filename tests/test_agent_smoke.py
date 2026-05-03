from __future__ import annotations

from explainable_agent.agent import ExplainableAgent, tool_support_score
from explainable_agent.config import Settings
from explainable_agent.schemas import Decision, StepTrace


class FakeClient:
    def resolve_model(self, requested_model: str) -> str:
        return requested_model

    def get_decision(self, **_kwargs):
        return (
            Decision(
                action="final_answer",
                rationale="Test final answer based on deterministic tool output.",
                confidence=1.0,
                evidence=["The deterministic calculate_math tool returned 848.0."],
                answer="848.0",
            ),
            '{"action":"final_answer","answer":"848.0"}',
            0,
            {},
        )

    def _provider_prefers_plain_decisions(self) -> bool:
        return True


def test_explicit_math_request_runs_without_llm(tmp_path):
    settings = Settings.from_env().with_overrides(
        requested_model="qwen3.5:9b",
        runs_dir=tmp_path / "runs",
        workspace_root=tmp_path,
        max_steps=2,
    )

    agent = ExplainableAgent(settings=settings, client=FakeClient(), verbose=False)
    trace = agent.run("calculate_math: (215*4)-12")

    assert trace.requested_model == "qwen3.5:9b"
    assert trace.final_answer == "848.0"
    assert trace.steps[0].decision.tool_name == "calculate_math"


def test_tool_support_score_matches_numeric_and_path_outputs():
    steps = [
        StepTrace(
            step=1,
            model_output="",
            decision=Decision(
                action="tool_call",
                rationale="",
                confidence=1.0,
                evidence=[],
                tool_name="calculate_math",
                tool_input="",
            ),
            tool_output="866.0\nexamples/basic_usage.py",
            latency_ms=0,
        )
    ]

    score = tool_support_score("Result: 866 from examples/basic_usage.py", steps)

    assert score >= 0.5


def test_tool_support_score_accepts_numeric_tool_answers():
    steps = [
        StepTrace(
            step=1,
            model_output="",
            decision=Decision(
                action="tool_call",
                rationale="",
                confidence=1.0,
                evidence=[],
                tool_name="calculate_math",
                tool_input="",
            ),
            tool_output="84.0",
            latency_ms=0,
        )
    ]

    assert tool_support_score("The result is 84.0", steps) > 0
