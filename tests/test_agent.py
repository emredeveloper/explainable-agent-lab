from explainable_agent.agent import (
    _build_step_audit,
    _extract_math_expression,
    _extract_sql_statement,
    _fallback_answer_from_tool_outputs,
    _heuristic_tool_suggestion,
    _is_low_quality_answer,
    _looks_generic_completion,
    _should_override_first_tool,
    tool_support_score,
)
from explainable_agent.schemas import Decision, StepTrace


def test_extract_math_expression() -> None:
    expr = _extract_math_expression("hesap: (215*4)-12")
    assert expr == "(215*4)-12"


def test_heuristic_tool_suggestion_explicit_math_tool() -> None:
    suggestion = _heuristic_tool_suggestion("calculate_math: 2^10")
    assert suggestion == ("calculate_math", "2^10")


def test_heuristic_tool_suggestion_from_sql() -> None:
    suggestion = _heuristic_tool_suggestion("SELECT name FROM customers;")
    assert suggestion == ("sqlite_query", "SELECT name FROM customers;")


def test_heuristic_tool_suggestion_from_glob() -> None:
    suggestion = _heuristic_tool_suggestion("*.py dosyalarini goster")
    assert suggestion == ("list_workspace_files", ".|*.py")


def test_extract_sql_statement() -> None:
    sql = _extract_sql_statement("Calistir: SELECT id, name FROM customers ORDER BY id;")
    assert sql == "SELECT id, name FROM customers ORDER BY id;"


def test_should_override_first_tool() -> None:
    assert _should_override_first_tool("calculate_math", "sqlite_query")
    assert not _should_override_first_tool("sqlite_query", "sqlite_query")


def test_is_low_quality_answer() -> None:
    assert _is_low_quality_answer("final_answer")
    assert _is_low_quality_answer('{"action":"final_answer"}')
    assert _is_low_quality_answer("sqlite_query")
    assert not _is_low_quality_answer("Toplam sonuc 848")


def test_looks_generic_completion() -> None:
    assert _looks_generic_completion("Done")
    assert _looks_generic_completion("Tamam")
    assert not _looks_generic_completion("Acme Corp | Istanbul")
    assert not _looks_generic_completion("Toplam: 848")


def test_fallback_answer_from_tool_outputs_file_list() -> None:
    steps = [
        StepTrace(
            step=1,
            model_output="{}",
            decision=Decision(
                action="tool_call",
                rationale="r",
                confidence=0.8,
                evidence=["e"],
                tool_name="list_workspace_files",
                tool_input=".|*.py",
            ),
            tool_output="a.py\nb.py\nc.py",
            latency_ms=100,
        )
    ]
    fallback = _fallback_answer_from_tool_outputs(steps)
    assert fallback is not None
    assert "a.py" in fallback


def test_fallback_answer_from_tool_outputs_db() -> None:
    steps = [
        StepTrace(
            step=1,
            model_output="{}",
            decision=Decision(
                action="tool_call",
                rationale="r",
                confidence=0.8,
                evidence=["e"],
                tool_name="sqlite_query",
                tool_input="SELECT name FROM customers;",
            ),
            tool_output="COLUMNS: name\nROWS:\nAcme Corp\nROW_COUNT: 1",
            latency_ms=100,
        )
    ]
    fallback = _fallback_answer_from_tool_outputs(steps)
    assert fallback is not None
    assert "SQLite result" in fallback


def test_tool_support_score() -> None:
    steps = [
        StepTrace(
            step=1,
            model_output="{}",
            decision=Decision(
                action="tool_call",
                rationale="r",
                confidence=0.8,
                evidence=["e"],
                tool_name="sqlite_query",
                tool_input="SELECT",
            ),
            tool_output="KOLONLAR: name\nSATIRLAR:\nAcme Corp",
            latency_ms=100,
        )
    ]
    score = tool_support_score("Musteri adi Acme Corp", steps)
    assert score > 0


def test_build_step_audit_warnings() -> None:
    decision = Decision(
        action="final_answer",
        rationale="r",
        confidence=0.2,
        evidence=["e"],
        answer="Tamam",
    )
    audit = _build_step_audit(decision=decision, source="model")
    assert audit["source"] == "model"
    assert any("Low confidence" in warning for warning in audit["warnings"])
