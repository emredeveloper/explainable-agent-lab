from scripts.eval_model_matrix import exact_value, grade_answer


def test_substring_match_is_not_correctness():
    assert not grade_answer('{"result": 1848}', {"result": 848})["exact_facts"]


def test_extra_claims_and_assumed_currency_fail():
    expected = {"total": 444, "currency": None}
    assert grade_answer('{"total":444,"currency":null}', expected)["exact_facts"]
    assert not grade_answer('{"total":444,"currency":"USD"}', expected)["exact_facts"]
    assert not grade_answer('{"total":444,"currency":null,"verified":true}', expected)[
        "exact_facts"
    ]


def test_schema_types_are_checked():
    assert not exact_value({"result": True}, {"result": 1})
    assert not exact_value({"result": "848"}, {"result": 848})
    assert exact_value({"result": 848.0}, {"result": 848})


def test_missing_fact_is_not_an_empty_substring_check():
    expected = {"supplier_email": None}
    assert not grade_answer('{"supplier_email":"made-up@example.com"}', expected)[
        "exact_facts"
    ]
    assert not grade_answer("{}", expected)["exact_facts"]


def test_invalid_json_and_nonfinite_values_fail():
    assert not grade_answer('```json\n{"result":848}\n```', {"result": 848})[
        "valid_json"
    ]
    assert not grade_answer('{"result":NaN}', {"result": 848})["exact_facts"]
