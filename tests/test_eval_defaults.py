from __future__ import annotations

from scripts.eval_hf_tool_calls import DEFAULT_DATASET


def test_hf_eval_default_dataset_exists():
    assert DEFAULT_DATASET.exists()
