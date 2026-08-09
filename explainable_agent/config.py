from __future__ import annotations

import os
from dataclasses import dataclass, replace
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
    chaos_mode: bool
    use_native_tools: bool = False
    stream: bool = False

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
        chaos_mode: bool | None = None,
        use_native_tools: bool | None = None,
        stream: bool | None = None,
    ) -> Settings:
        return replace(
            self,
            base_url=base_url if base_url is not None else self.base_url,
            api_key=api_key if api_key is not None else self.api_key,
            requested_model=(
                requested_model if requested_model is not None else self.requested_model
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
            chaos_mode=chaos_mode if chaos_mode is not None else self.chaos_mode,
            use_native_tools=use_native_tools
            if use_native_tools is not None
            else self.use_native_tools,
            stream=stream if stream is not None else self.stream,
        )

    @classmethod
    def from_env(cls) -> Settings:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    if key and key not in os.environ:
                        os.environ[key] = _strip_quotes(val.strip())

        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "local"),
            requested_model=os.getenv("AGENT_MODEL", "gpt-oss-20b"),
            reasoning_effort=os.getenv("AGENT_REASONING_EFFORT", "high"),
            max_steps=_env_int("AGENT_MAX_STEPS", 6),
            runs_dir=Path(os.getenv("AGENT_RUNS_DIR", "runs")).resolve(),
            workspace_root=Path(os.getenv("AGENT_WORKSPACE", ".")).resolve(),
            temperature=_env_float("AGENT_TEMPERATURE", 0.2),
            chaos_mode=_env_bool("AGENT_CHAOS_MODE"),
            use_native_tools=_env_bool("AGENT_NATIVE_TOOLS"),
            stream=_env_bool("AGENT_STREAM"),
        )


def _strip_quotes(value: str) -> str:
    """Drop one layer of matching quotes, as dotenv-style parsers do."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _env_bool(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"true", "1", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for {name}: {raw!r}. Expected a whole number."
        ) from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid number for {name}: {raw!r}. Expected a decimal number."
        ) from exc
