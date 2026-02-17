from scripts.eval_hf_tool_calls import _build_actionable_plan


def test_build_actionable_plan_returns_concise_items() -> None:
    plan = _build_actionable_plan(
        summary_metrics={
            "name_match_accuracy": 0.6,
            "call_count_accuracy": 0.7,
            "argument_match_rate": 0.5,
        },
        failure_patterns=[
            {
                "error_type": "wrong_tool_name",
                "name_match": False,
                "call_count_match": True,
                "parse_error": False,
                "count": 4,
            }
        ],
        guard_metrics={
            "dropped_by_max_tool_calls": 2,
            "missing_required_keys": 3,
            "schema_validation_errors": 1,
        },
    )
    assert len(plan) <= 3
    assert len(plan) >= 1
    assert any("pattern" in item.lower() for item in plan)
