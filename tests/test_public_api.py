from __future__ import annotations

from pathlib import Path

import explainable_agent
from explainable_agent import (
    Decision,
    ExplainableAgent,
    Settings,
    ToolRegistry,
    ToolSpec,
    define_tool,
    run_tool,
    write_run_artifacts,
)


class FakeClient:
    def resolve_model(self, requested_model: str) -> str:
        return requested_model

    def get_decision(self, **_kwargs):
        return (
            Decision(
                action="final_answer",
                rationale="Test final answer based on deterministic tool output.",
                confidence=1.0,
                evidence=["The deterministic tool returned local:ok."],
                answer="local:ok",
            ),
            '{"action":"final_answer","answer":"local:ok"}',
            0,
            {},
        )

    def _provider_prefers_plain_decisions(self) -> bool:
        return True


def test_public_api_exports_library_surface():
    assert explainable_agent.__version__ == "0.3.1"
    assert Decision is not None
    assert ExplainableAgent is not None
    assert Settings is not None
    assert ToolRegistry is not None
    assert ToolSpec is not None
    assert define_tool is not None
    assert run_tool is not None
    assert write_run_artifacts is not None


def test_tool_registry_can_isolate_custom_tools(tmp_path):
    registry = ToolRegistry()

    @registry.define_tool(
        name="local_echo",
        description="Echoes text.",
        usage_hint="Input is plain text.",
    )
    def local_echo(text: str, _workspace_root: Path) -> str:
        return f"local:{text}"

    assert registry.run("local_echo", "ok", tmp_path) == "local:ok"
    assert run_tool("local_echo", "ok", tmp_path).startswith("ERROR: unknown tool")


def test_agent_uses_isolated_tool_registry(tmp_path):
    registry = ToolRegistry()

    @registry.define_tool(
        name="local_echo",
        description="Echoes text.",
        usage_hint="Input is plain text.",
    )
    def local_echo(text: str, _workspace_root: Path) -> str:
        return f"local:{text}"

    settings = Settings.from_env().with_overrides(
        requested_model="test-model",
        runs_dir=tmp_path / "runs",
        workspace_root=tmp_path,
        max_steps=2,
    )
    agent = ExplainableAgent(
        settings=settings,
        client=FakeClient(),
        tool_registry=registry,
    )

    trace = agent.run("local_echo: ok")

    assert trace.steps[0].decision.tool_name == "local_echo"
    assert trace.steps[0].tool_output == "local:ok"
    assert trace.final_answer == "local:ok"
