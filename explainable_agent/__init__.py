"""Public API for the explainable-agent package."""

from .agent import ExplainableAgent
from .config import Settings
from .openai_client import LLMConnectionError
from .report import write_orchestrator_artifacts, write_run_artifacts
from .schemas import (
    Decision,
    FaithfulnessCheck,
    OrchestratorRunTrace,
    RunTrace,
    StepTrace,
    SubTaskTrace,
)
from .tools import ToolRegistry, ToolSpec, define_tool, run_tool

__version__ = "0.3.1"

__all__ = [
    "Decision",
    "ExplainableAgent",
    "FaithfulnessCheck",
    "LLMConnectionError",
    "OrchestratorRunTrace",
    "RunTrace",
    "Settings",
    "StepTrace",
    "SubTaskTrace",
    "ToolRegistry",
    "ToolSpec",
    "__version__",
    "define_tool",
    "run_tool",
    "write_orchestrator_artifacts",
    "write_run_artifacts",
]
