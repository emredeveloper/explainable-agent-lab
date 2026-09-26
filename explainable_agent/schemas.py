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
    model_output_length: int = 0
    tool_output_length: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
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
            "model_output_length": self.model_output_length,
            "tool_output_length": self.tool_output_length,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
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
    efficiency_diagnostics: list[str] = field(default_factory=list)
    llm_usage: dict[str, int] | None = None

    @property
    def total_usage(self) -> dict[str, int]:
        if self.llm_usage is not None:
            return dict(self.llm_usage)
        return {
            key: sum(getattr(step, key) for step in self.steps)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    @property
    def recovery_counts(self) -> dict[str, int]:
        attempts = 0
        successes = 0
        pending_error = False
        for step in self.steps:
            if step.decision.action != "tool_call" or step.tool_output is None:
                continue
            failed = step.tool_output.startswith("ERROR:")
            if pending_error:
                attempts += 1
                if not failed:
                    successes += 1
            pending_error = failed
        return {"retry_attempts": attempts, "successful_retries": successes}

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
            "efficiency_diagnostics": list(self.efficiency_diagnostics),
            "recovery_counts": self.recovery_counts,
            "llm_usage": self.total_usage,
        }


@dataclass
class SubTaskTrace:
    agent_name: str
    assigned_task: str
    orchestrator_rationale: str
    trace: RunTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "assigned_task": self.assigned_task,
            "orchestrator_rationale": self.orchestrator_rationale,
            "trace": self.trace.to_dict(),
        }


@dataclass
class OrchestratorRunTrace:
    run_id: str
    main_task: str
    started_at_utc: str
    finished_at_utc: str
    subtasks: list[SubTaskTrace]
    final_synthesis: str
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "main_task": self.main_task,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "final_synthesis": self.final_synthesis,
            "diagnostics": list(self.diagnostics),
        }
