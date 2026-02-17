from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    requested_model: str
    reasoning_effort: str
    max_steps: int
    runs_dir: Path
    workspace_root: Path
    temperature: float

    def with_overrides(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        requested_model: str | None = None,
        reasoning_effort: str | None = None,
        max_steps: int | None = None,
        runs_dir: Path | None = None,
        workspace_root: Path | None = None,
        temperature: float | None = None,
    ) -> "Settings":
        return replace(
            self,
            base_url=base_url if base_url is not None else self.base_url,
            api_key=api_key if api_key is not None else self.api_key,
            requested_model=(
                requested_model
                if requested_model is not None
                else self.requested_model
            ),
            reasoning_effort=(
                reasoning_effort
                if reasoning_effort is not None
                else self.reasoning_effort
            ),
            max_steps=max_steps if max_steps is not None else self.max_steps,
            runs_dir=runs_dir if runs_dir is not None else self.runs_dir,
            workspace_root=(
                workspace_root if workspace_root is not None else self.workspace_root
            ),
            temperature=temperature if temperature is not None else self.temperature,
        )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            requested_model=os.getenv("AGENT_MODEL", "gpt-oss-20b"),
            reasoning_effort=os.getenv("AGENT_REASONING_EFFORT", "high"),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
            runs_dir=Path(os.getenv("AGENT_RUNS_DIR", "runs")).resolve(),
            workspace_root=Path(os.getenv("AGENT_WORKSPACE", ".")).resolve(),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
        )
