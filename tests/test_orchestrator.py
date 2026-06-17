from __future__ import annotations

from explainable_agent.orchestrator import TeamOrchestrator
from explainable_agent.schemas import Decision, FaithfulnessCheck, RunTrace, StepTrace


def _trace(answer: str = "done") -> RunTrace:
    return RunTrace(
        run_id="test-run",
        task="subtask",
        requested_model="test-model",
        resolved_model="test-model",
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
        steps=[
            StepTrace(
                step=1,
                model_output="{}",
                decision=Decision(
                    action="final_answer",
                    rationale="done",
                    confidence=1.0,
                    evidence=["test"],
                    answer=answer,
                ),
                tool_output=None,
                latency_ms=0,
            )
        ],
        final_answer=answer,
        faithfulness=FaithfulnessCheck(
            alternative_answer="",
            lexical_similarity=1.0,
            threshold=0.75,
            likely_faithful=False,
            note="test",
        ),
    )


class FakePlanClient:
    def __init__(self, payload):
        self.payload = payload
        self.synthesis_calls = 0

    def resolve_model(self, requested_model: str) -> str:
        return requested_model

    def get_json_object(self, **_kwargs):
        return self.payload, "{}", 0, {}

    def get_alternative_answer(self, **_kwargs):
        self.synthesis_calls += 1
        return "synthesized"


class FakeAgent:
    def run(self, _task: str) -> RunTrace:
        return _trace("subtask complete")


def test_orchestrator_does_not_synthesize_empty_plan():
    client = FakePlanClient({"plan": []})
    orchestrator = TeamOrchestrator(
        client=client,
        agents={"worker": ("Does test work", FakeAgent())},
    )

    trace = orchestrator.run("main task", requested_model="test-model")

    assert trace.subtasks == []
    assert "No sub-agent tasks were executed" in trace.final_synthesis
    assert client.synthesis_calls == 0


def test_orchestrator_runs_valid_plan_and_synthesizes():
    client = FakePlanClient(
        {
            "plan": [
                {
                    "agent_name": "worker",
                    "assigned_task": "do the work",
                    "rationale": "best fit",
                }
            ]
        }
    )
    orchestrator = TeamOrchestrator(
        client=client,
        agents={"worker": ("Does test work", FakeAgent())},
    )

    trace = orchestrator.run("main task", requested_model="test-model")

    assert len(trace.subtasks) == 1
    assert trace.final_synthesis == "synthesized"
    assert client.synthesis_calls == 1
