"""Regression cases discovered while exercising a local Ollama model."""

from types import SimpleNamespace

import pytest
from openai import APIConnectionError

from explainable_agent import ExplainableAgent, Settings, ToolRegistry
from explainable_agent.agent import _unsupported_currency_symbols, tool_support_score
from explainable_agent.json_utils import parse_text_tool_call
from explainable_agent.openai_client import LLMConnectionError, OpenAICompatClient
from explainable_agent.report import _generate_diagnostics, _to_markdown_report
from explainable_agent.schemas import Decision, FaithfulnessCheck, RunTrace, StepTrace
from explainable_agent.tools import available_tool_names, run_tool


def tool_step(output, analysis=None):
    return StepTrace(
        step=1,
        model_output="",
        latency_ms=0,
        tool_output=output,
        decision=Decision(
            action="tool_call",
            rationale="test",
            confidence=0.5,
            evidence=[],
            tool_name="lookup",
            error_analysis=analysis,
        ),
    )


def trace_with(steps):
    return RunTrace(
        run_id="test",
        task="test",
        requested_model="m",
        resolved_model="m",
        started_at_utc="2026-09-05T00:00:00+00:00",
        finished_at_utc="2026-09-05T00:00:01+00:00",
        steps=steps,
        final_answer="999",
        faithfulness=FaithfulnessCheck("", 0, 0.75, False, "test"),
    )


@pytest.mark.parametrize(
    ("raw", "tool", "value"),
    [
        (
            "<|tool_call_start|>[calculate_math(expression='37*12')]<|tool_call_end|>",
            "calculate_math",
            "37*12",
        ),
        (
            "I will call the reader.\n<tool_call>\n<function=tool>\n"
            "<function=tool_name>\nflaky_lookup\n</function>\n"
            "<function=tool_input>\n\n</function>\n</tool_call>",
            "flaky_lookup",
            "",
        ),
        (
            'I have the data.\ntool_name: "calculate_math"\n'
            'tool_input: "(37*12)"',
            "calculate_math",
            "(37*12)",
        ),
        (
            "<tool_call><function=tool><parameter=action>tool_call</parameter>"
            "<parameter=tool_name>flaky_lookup</parameter>"
            "<parameter=tool_input>{}</parameter></function></tool_call>",
            "flaky_lookup",
            "",
        ),
        (
            'tool_name="broken_lookup"\ntool_input={}',
            "broken_lookup",
            "",
        ),
        (
            "<|tool_call_start|>[read_text_file(file_path='inventory.txt')]"
            "<|tool_call_end|>",
            "read_text_file",
            "inventory.txt",
        ),
    ],
)
def test_supported_text_tool_calls_are_decisions(raw, tool, value):
    client = OpenAICompatClient
    decision = client._to_decision(client._parse_json_payload(raw), raw)
    assert (decision.action, decision.tool_name, decision.tool_input) == (
        "tool_call",
        tool,
        value,
    )


@pytest.mark.parametrize(
    "body",
    [
        "[lookup(__import__('os').getcwd())]",
        "[obj.lookup('x')]",
        "[lookup(**{'input': 'x'})]",
        "[lookup('x'), lookup('y')]",
        "[lookup(a='x', b='y')]",
    ],
)
def test_tagged_protocol_does_not_evaluate_code_or_drop_arguments(body):
    assert parse_text_tool_call(f"<|tool_call_start|>{body}<|tool_call_end|>") is None


def test_unsupported_protocol_is_not_presented_as_success():
    raw = "<|tool_call_start|>[lookup(a='x', b='y')]<|tool_call_end|>"
    payload = OpenAICompatClient._parse_json_payload(raw)
    assert payload["answer"].startswith("ERROR:")
    assert payload["confidence"] == 0


def test_malformed_xml_tool_call_is_not_presented_as_success():
    raw = "<tool_call><function=tool_name>lookup</function></tool_call>"
    payload = OpenAICompatClient._parse_json_payload(raw)
    assert payload["answer"].startswith("ERROR:")


@pytest.mark.parametrize(
    "raw",
    [
        "action: final_answer",
        "action: tool_call\ntool_name: lookup",
        "action: unknown\nanswer: maybe",
    ],
)
def test_incomplete_field_decisions_are_not_presented_as_answers(raw):
    payload = OpenAICompatClient._parse_json_payload(raw)
    assert payload["answer"].startswith("ERROR:")
    assert payload["confidence"] == 0


@pytest.mark.parametrize(
    ("raw", "answer"),
    [
        (
            "A short preamble.\naction: final_answer\n"
            "answer: The total value is 444.\n",
            "The total value is 444.",
        ),
        ("answer: The exact code is AMBER-7391.", "The exact code is AMBER-7391."),
    ],
)
def test_field_based_final_answers_are_decisions(raw, answer):
    decision = OpenAICompatClient._to_decision(
        OpenAICompatClient._parse_json_payload(raw), raw
    )
    assert decision.action == "final_answer"
    assert decision.answer == answer


def test_unclosed_xml_tool_call_with_complete_fields_is_decoded():
    raw = (
        "<tool_call><function=tool><parameter=action>tool_call</parameter>"
        "<parameter=tool_name>flaky_lookup</parameter>"
        "<parameter=tool_input>{}</parameter></function>"
    )
    decision = OpenAICompatClient._to_decision(
        OpenAICompatClient._parse_json_payload(raw), raw
    )
    assert (decision.action, decision.tool_name, decision.tool_input) == (
        "tool_call",
        "flaky_lookup",
        "",
    )


def test_xml_final_answer_fields_are_normalized():
    raw = (
        "<tool_call><function=tool><parameter=action>final_answer</parameter>"
        "<parameter=answer>The warehouse code is AMBER-7391.</parameter>"
        "<parameter=confidence>0.99</parameter></function>"
    )
    decision = OpenAICompatClient._to_decision(
        OpenAICompatClient._parse_json_payload(raw), raw
    )
    assert decision.action == "final_answer"
    assert decision.answer == "The warehouse code is AMBER-7391."


def test_numeric_overlap_does_not_match_only_decimal_zero():
    assert tool_support_score("The answer is 999.0", [tool_step("848.0")]) == 0
    assert tool_support_score("7", [tool_step("7.0")]) == 1
    assert tool_support_score("7", [tool_step("-7")]) == 0


def test_error_outputs_do_not_count_as_support():
    assert tool_support_score("file missing", [tool_step("ERROR: file missing")]) == 0


def test_currency_must_come_from_task_or_successful_result():
    assert _unsupported_currency_symbols("total price", "$444", [tool_step("444")]) == {
        "$"
    }
    assert (
        _unsupported_currency_symbols("total in $", "$444", [tool_step("444")]) == set()
    )
    assert _unsupported_currency_symbols("total", "$444", [tool_step("$444")]) == set()


def test_separate_context_does_not_trigger_explicit_tool_input(tmp_path):
    class Client:
        def resolve_model(self, model):
            return model

        def _provider_prefers_plain_decisions(self):
            return True

        def get_decision(self, **kwargs):
            assert any(
                "Previous result: 7391" in m["content"] for m in kwargs["messages"]
            )
            return Decision("final_answer", "test", 1, [], answer="2.0"), "{}", 0, {}

    trace = ExplainableAgent(
        Settings.from_env().with_overrides(workspace_root=tmp_path, max_steps=2),
        client=Client(),
    ).run("calculate_math: 1+1", context="Previous result: 7391")
    assert trace.steps[0].tool_output == "2.0"
    assert trace.task == "calculate_math: 1+1"


def test_recovery_counts_observed_outcome_without_model_analysis():
    trace = trace_with(
        [
            tool_step("ERROR: timeout"),
            tool_step("ERROR: timeout", "retry"),
            tool_step("found"),
        ]
    )
    assert trace.recovery_counts == {"retry_attempts": 2, "successful_retries": 1}
    trace.steps.pop()
    assert trace.recovery_counts == {"retry_attempts": 1, "successful_retries": 0}


def test_zero_support_produces_report_warning():
    assert any(
        "FAITHFULNESS WARNING" in text
        for text in _generate_diagnostics(trace_with([tool_step("848")]))
    )


def test_empty_registry_does_not_enable_global_tools(tmp_path):
    registry = ToolRegistry()
    assert available_tool_names(registry) == set()
    assert available_tool_names({}) == set()
    assert run_tool("calculate_math", "1+1", tmp_path, registry).startswith(
        "ERROR: unknown tool"
    )


def test_first_step_heuristic_cannot_add_a_disabled_tool(tmp_path):
    agent = ExplainableAgent(
        Settings.from_env().with_overrides(workspace_root=tmp_path),
        tool_registry=ToolRegistry(),
    )
    decision = Decision(
        "final_answer",
        "No tools available",
        0.5,
        [],
        answer="I have no calculation tool.",
    )
    actual, _, source, _ = agent._apply_first_step_heuristics(
        task="compute 37*12", steps=[], decision=decision, raw_output="{}"
    )
    assert actual is decision
    assert source == "model"


def test_custom_tool_exception_becomes_recoverable_result(tmp_path):
    registry = ToolRegistry()

    @registry.define_tool("lookup", "test", "test")
    def lookup(text, workspace):
        raise TimeoutError("try again")

    assert registry.run("lookup", "", tmp_path) == "ERROR: TimeoutError: try again"


def test_stream_requests_usage_and_preserves_connection_failure(monkeypatch):
    client = OpenAICompatClient("http://localhost:11434/v1", "local")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise APIConnectionError(request=None)

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    with pytest.raises(LLMConnectionError):
        client.get_decision_stream("m", [], 0, "low")
    assert len(calls) == 1
    assert calls[0]["stream_options"] == {"include_usage": True}


def test_stream_usage_only_chunk_is_counted(monkeypatch):
    client = OpenAICompatClient("http://localhost:11434/v1", "local")
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content='{"action":"final_answer","answer":"42"}'
                    )
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        ),
    ]
    monkeypatch.setattr(
        client.client.chat.completions, "create", lambda **kw: iter(chunks)
    )
    decision, _, _, usage = client.get_decision_stream("m", [], 0, "low")
    assert decision.answer == "42"
    assert (
        usage
        == client.usage_totals
        == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )


def test_wrong_answer_is_not_faithful_and_auxiliary_usage_is_counted(
    monkeypatch, tmp_path
):
    client = OpenAICompatClient("http://localhost:1234/v1", "local")
    monkeypatch.setattr(client, "resolve_model", lambda name: name)
    responses = iter(['{"action":"final_answer","answer":"999.0"}', "848.0"] * 2)

    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    agent = ExplainableAgent(
        Settings.from_env().with_overrides(workspace_root=tmp_path, max_steps=2),
        client=client,
    )
    for _ in range(2):
        trace = agent.run("calculate_math: (215*4)-12")
        assert not trace.faithfulness.likely_faithful
        assert trace.faithfulness.tool_support_score == 0
        assert sum(step.total_tokens for step in trace.steps) == 15
        assert trace.total_usage["total_tokens"] == 30
        assert "| Total tokens | `30` |" in _to_markdown_report(trace)
