from explainable_agent.json_utils import parse_json_object_relaxed


def test_parse_json_object_relaxed_direct_json() -> None:
    payload, method = parse_json_object_relaxed('{"a":1,"b":"x"}')
    assert payload == {"a": 1, "b": "x"}
    assert method is None


def test_parse_json_object_relaxed_with_noise_and_missing_bracket() -> None:
    raw = 'text {"analysis":{"root":"x"},"files_to_inspect":["a.py"],"first_actions":["do this"}'
    payload, method = parse_json_object_relaxed(raw)
    assert payload is not None
    assert "analysis" in payload
    assert method in {"json_repair", "extracted", None}
