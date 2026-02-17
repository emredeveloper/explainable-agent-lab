import json
from pathlib import Path

from explainable_agent.report import write_run_artifacts
from explainable_agent.schemas import Decision, FaithfulnessCheck, RunTrace, StepTrace


def test_write_run_artifacts(tmp_path: Path) -> None:
    trace = RunTrace(
        run_id="run_test",
        task="test task",
        requested_model="gpt-oss-20b",
        resolved_model="gpt-oss-20b",
        started_at_utc="2026-02-15T00:00:00+00:00",
        finished_at_utc="2026-02-15T00:00:01+00:00",
        steps=[
            StepTrace(
                step=1,
                model_output='{"action":"final_answer"}',
                decision=Decision(
                    action="final_answer",
                    rationale="enough context",
                    confidence=0.8,
                    evidence=["signal1"],
                    answer="done",
                ),
                tool_output=None,
                latency_ms=120,
            )
        ],
        final_answer="done",
        faithfulness=FaithfulnessCheck(
            alternative_answer="done",
            lexical_similarity=1.0,
            threshold=0.75,
            likely_faithful=False,
            note="test",
        ),
    )

    trace_path, report_path = write_run_artifacts(trace, tmp_path)
    assert trace_path.exists()
    assert report_path.exists()
    full_trace_path = tmp_path / "run_test" / "trace_full.json"
    assert full_trace_path.exists()

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["trace_version"] == "compact-v2"
    assert payload["steps"][0]["action"] == "final_answer"
    assert "model_output" not in json.dumps(payload)
    assert "Aciklanabilir Ajan Calisma Raporu" in report_path.read_text(encoding="utf-8")
