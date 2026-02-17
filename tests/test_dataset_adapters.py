from pathlib import Path

from explainable_agent.dataset_adapters import (
    load_dataset_with_adapter,
    load_swebench_lite_samples,
    resolve_dataset_format,
)


def test_resolve_dataset_format_swebench() -> None:
    fmt = resolve_dataset_format(Path("swebench_lite_sample.jsonl"), "auto")
    assert fmt == "swebench_lite"


def test_load_swebench_lite_samples_jsonl(tmp_path: Path) -> None:
    sample = tmp_path / "swebench_lite_sample.jsonl"
    sample.write_text(
        '{"instance_id":"x1","repo":"org/repo","base_commit":"abc","problem_statement":"Fix failing parser."}\n',
        encoding="utf-8",
    )
    rows = load_swebench_lite_samples(sample, limit=10)
    assert len(rows) == 1
    assert rows[0]["task_type"] == "swebench_patch"
    assert rows[0]["source_row_id"] == "x1"


def test_load_dataset_with_adapter_swebench(tmp_path: Path) -> None:
    sample = tmp_path / "swebench_lite_sample.jsonl"
    sample.write_text(
        '{"instance_id":"x2","problem_statement":"Fix bug in cache invalidation."}\n',
        encoding="utf-8",
    )
    output = load_dataset_with_adapter(
        dataset_path=sample,
        dataset_format="swebench_lite",
        limit=5,
    )
    assert output.task_type == "swebench_patch"
    assert len(output.rows) == 1
