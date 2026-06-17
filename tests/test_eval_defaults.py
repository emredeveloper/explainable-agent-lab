from __future__ import annotations

from pathlib import Path


def test_hf_eval_default_dataset_exists():
    expected_path = "data/evals/hf_complexfuncbench_first_turn_100.jsonl"
    script_text = Path("scripts/eval_hf_tool_calls.py").read_text(encoding="utf-8")

    assert f'DEFAULT_DATASET = Path("{expected_path}")' in script_text
    assert Path(expected_path).exists()
