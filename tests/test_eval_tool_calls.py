from explainable_agent.eval_tool_calls import (
    load_bfcl_sql_samples,
    normalize_tool_calls,
    score_prediction,
    score_prediction_variants,
)


def test_normalize_tool_calls_parses_argument_string() -> None:
    calls = [{"name": "foo", "arguments": '{"a":2,"b":1}'}]
    normalized = normalize_tool_calls(calls)
    assert normalized == [{"name": "foo", "arguments": {"a": 2, "b": 1}}]


def test_score_prediction_exact_match() -> None:
    expected = [
        {"name": "foo", "arguments": {"x": 1}},
        {"name": "bar", "arguments": {"y": "z"}},
    ]
    predicted = [
        {"name": "foo", "arguments": {"x": 1}},
        {"name": "bar", "arguments": {"y": "z"}},
    ]
    exact, names_ok, count_ok, arg_count, err = score_prediction(
        expected_calls=expected,
        predicted_calls=predicted,
        parse_error=False,
    )
    assert exact
    assert names_ok
    assert count_ok
    assert arg_count == 2
    assert err is None


def test_score_prediction_wrong_name() -> None:
    expected = [{"name": "foo", "arguments": {"x": 1}}]
    predicted = [{"name": "bar", "arguments": {"x": 1}}]
    exact, names_ok, count_ok, arg_count, err = score_prediction(
        expected_calls=expected,
        predicted_calls=predicted,
        parse_error=False,
    )
    assert not exact
    assert not names_ok
    assert count_ok
    assert arg_count == 1
    assert err == "wrong_tool_name"


def test_score_prediction_variants_selects_best_match() -> None:
    expected_variants = [
        [{"name": "foo", "arguments": {"x": 1}}],
        [{"name": "bar", "arguments": {"y": 2}}],
    ]
    predicted = [{"name": "bar", "arguments": {"y": 2}}]

    exact, names_ok, count_ok, arg_count, err, variant_index = score_prediction_variants(
        expected_variants=expected_variants,
        predicted_calls=predicted,
        parse_error=False,
    )

    assert exact
    assert names_ok
    assert count_ok
    assert arg_count == 1
    assert err is None
    assert variant_index == 1


def test_load_bfcl_sql_samples_normalizes_rows(tmp_path) -> None:
    question_path = tmp_path / "BFCL_v3_sql.json"
    answer_dir = tmp_path / "possible_answer"
    answer_dir.mkdir()
    answer_path = answer_dir / "BFCL_v3_sql.json"

    question_path.write_text(
        '{"id":"sql_1","question":[{"role":"user","content":"Find student name by id."}],"function":[{"name":"sql.execute","description":"Run SQL","parameters":{"type":"dict","properties":{"table_name":{"type":"string"}}}}]}\n',
        encoding="utf-8",
    )
    answer_path.write_text(
        '{"id":"sql_1","ground_truth":[{"sql.execute":{"sql_keyword":["SELECT"],"table_name":["Students"],"columns":[["Name"]],"conditions":[["id = 1"]]}}]}\n',
        encoding="utf-8",
    )

    samples = load_bfcl_sql_samples(question_path=question_path, answer_path=answer_path)
    assert len(samples) == 1

    sample = samples[0]
    assert sample["query"] == "Find student name by id."
    assert sample["tools"][0]["function"]["parameters"]["type"] == "object"

    expected_args = sample["expected_tool_calls"][0]["arguments"]
    assert expected_args["sql_keyword"] == "SELECT"
    assert expected_args["table_name"] == "students"
    assert expected_args["columns"] == ["name"]
    assert expected_args["conditions"] == ["id=1"]


def test_score_prediction_bfcl_delete_allows_missing_columns() -> None:
    expected = [
        {
            "name": "sql.execute",
            "arguments": {
                "sql_keyword": "DELETE",
                "table_name": "Students",
                "columns": ["StudentID", "Name", "GPA"],
                "conditions": ["GPA < 2.0"],
            },
        }
    ]
    predicted = [
        {
            "name": "sql.execute",
            "arguments": {
                "sql_keyword": "DELETE",
                "table_name": "students",
                "conditions": ["gpa<2.0"],
            },
        }
    ]
    exact, names_ok, count_ok, arg_count, err = score_prediction(
        expected_calls=expected,
        predicted_calls=predicted,
        parse_error=False,
    )
    assert exact
    assert names_ok
    assert count_ok
    assert arg_count == 1
    assert err is None


def test_score_prediction_bfcl_insert_accepts_single_variant_row() -> None:
    expected = [
        {
            "name": "sql.execute",
            "arguments": {
                "sql_keyword": "INSERT",
                "table_name": "mathscores",
                "columns": ["studentid", "name", "testscore", "testdate"],
                "insert_values": [
                    ["EW123", "Emily Watson", 95, "2022-03-01"],
                    ["EW123", "Emily Watson", 95, "03/01/2022"],
                    ["EW123", "Emily Watson", 95, "Mar 1, 2022"],
                ],
            },
        }
    ]
    predicted = [
        {
            "name": "sql.execute",
            "arguments": {
                "sql_keyword": "INSERT",
                "table_name": "MathScores",
                "columns": ["StudentID", "Name", "TestScore", "TestDate"],
                "insert_values": [["EW123", "Emily Watson", "95", "2022-03-01"]],
                "conditions": [],
                "update_values": None,
            },
        }
    ]
    exact, names_ok, count_ok, arg_count, err = score_prediction(
        expected_calls=expected,
        predicted_calls=predicted,
        parse_error=False,
    )
    assert exact
    assert names_ok
    assert count_ok
    assert arg_count == 1
    assert err is None
