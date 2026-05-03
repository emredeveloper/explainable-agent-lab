from __future__ import annotations

from pathlib import Path

from explainable_agent.config import Settings
from explainable_agent.tools import run_tool


def test_settings_loads_local_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=http://localhost:11434/v1",
                "OPENAI_API_KEY=ollama",
                "AGENT_MODEL=qwen3.5:9b",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env()

    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.api_key == "ollama"
    assert settings.requested_model == "qwen3.5:9b"


def test_calculate_math_tool_is_deterministic():
    assert run_tool("calculate_math", "(215*4)-12", Path.cwd()) == "848.0"


def test_file_tool_cannot_escape_workspace(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    result = run_tool("read_text_file", "../outside.txt", tmp_path)

    assert result.startswith("ERROR:")
    assert "escapes workspace" in result
