from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


ActionType = Literal["tool_call", "final_answer"]


@dataclass
class Decision:
    action: ActionType
    rationale: str
    confidence: float
    evidence: list[str]
    tool_name: str | None = None
    tool_input: str | None = None
    answer: str | None = None
    error_analysis: str | None = None
    proposed_fix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepTrace:
    step: int
    model_output: str
    decision: Decision
    tool_output: str | None
    latency_ms: int
    audit: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model_output": self.model_output,
            "decision": self.decision.to_dict(),
            "tool_output": self.tool_output,
            "latency_ms": self.latency_ms,
            "audit": dict(self.audit),
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class FaithfulnessCheck:
    alternative_answer: str
    lexical_similarity: float
    threshold: float
    likely_faithful: bool
    note: str
    tool_support_score: float = 0.0
    support_threshold: float = 0.25
    method: str = "lexical_jaccard+tool_overlap"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunTrace:
    run_id: str
    task: str
    requested_model: str
    resolved_model: str
    started_at_utc: str
    finished_at_utc: str
    steps: list[StepTrace]
    final_answer: str
    faithfulness: FaithfulnessCheck
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "faithfulness": self.faithfulness.to_dict(),
            "errors": list(self.errors),
        }
