import pytest

from src.assessment.evaluation_service import evaluate_code, select_tests
from src.assessment.question_service import load_question_bank
from src.utils.config import get_settings


@pytest.fixture(scope="module")
def bank():
    return {q["id"]: q for q in load_question_bank(get_settings().questions_dir)}


def reference_code(question: dict) -> str:
    return "\n".join(question["reference_solution"]) + "\n"


def test_correct_solution_passes_judge_mode(bank):
    q = bank["ARR-E-001"]
    ev = evaluate_code(reference_code(q), q["test_cases"], mode="judge", enable_execution=True)
    assert ev.status == "evaluated"
    assert ev.is_correct
    assert ev.passed == ev.total == len(q["test_cases"])
    assert ev.avg_duration_ms is not None


def test_sample_mode_runs_only_visible_tests(bank):
    q = bank["ARR-E-001"]
    visible = [t for t in q["test_cases"] if not t["hidden"]]
    ev = evaluate_code(reference_code(q), q["test_cases"], mode="sample", enable_execution=True)
    assert ev.total == len(visible)
    assert all(not o.hidden for o in ev.outcomes)


def test_wrong_answer_detected(bank):
    q = bank["ARR-E-001"]
    ev = evaluate_code("print(0)", q["test_cases"], mode="judge", enable_execution=True)
    assert not ev.is_correct
    assert any(o.status == "wrong_answer" for o in ev.outcomes)


def test_hidden_outcomes_do_not_leak_details(bank):
    q = bank["ARR-E-001"]
    ev = evaluate_code("print(0)", q["test_cases"], mode="judge", enable_execution=True)
    hidden = [o for o in ev.outcomes if o.hidden]
    assert hidden
    for o in hidden:
        assert o.input_text is None and o.expected is None
        assert o.actual is None and o.error is None
    visible = [o for o in ev.outcomes if not o.hidden]
    assert all(o.expected is not None for o in visible)


def test_syntax_error_reported_and_counts_as_attempt(bank):
    q = bank["ARR-E-001"]
    ev = evaluate_code("def f(:\n    pass\n", q["test_cases"], mode="judge", enable_execution=True)
    assert ev.status == "syntax_error"
    assert "line" in ev.message
    assert ev.counts_as_attempt and not ev.is_correct and ev.passed == 0


def test_runtime_error_summary_shown_for_visible_tests(bank):
    q = bank["ARR-E-001"]
    ev = evaluate_code("print(1 / 0)", q["test_cases"], mode="sample", enable_execution=True)
    assert ev.outcomes[0].status == "runtime_error"
    assert "ZeroDivisionError" in ev.outcomes[0].error


def test_infinite_loop_is_a_timeout(bank):
    q = bank["ARR-E-001"]
    ev = evaluate_code(
        "while True:\n    pass", q["test_cases"], mode="sample",
        enable_execution=True, timeout=1.0,
    )
    assert all(o.status == "timeout" for o in ev.outcomes)
    assert not ev.is_correct


def test_whitespace_differences_are_tolerated():
    cases = [{"input": "", "output": "hi", "hidden": False}]
    ev = evaluate_code("print('hi   ')\nprint()", cases, mode="sample", enable_execution=True)
    assert ev.is_correct


def test_execution_disabled_runs_nothing(bank):
    q = bank["ARR-E-001"]
    ev = evaluate_code(reference_code(q), q["test_cases"], mode="judge", enable_execution=False)
    assert ev.status == "disabled"
    assert not ev.counts_as_attempt and ev.outcomes == ()


def test_disabled_flag_read_from_settings(bank, monkeypatch):
    monkeypatch.setenv("ENABLE_CODE_EXECUTION", "false")
    q = bank["ARR-E-001"]
    assert evaluate_code("print(1)", q["test_cases"], mode="sample").status == "disabled"


def test_empty_and_overlong_code_invalid(bank):
    q = bank["ARR-E-001"]
    assert evaluate_code("   \n", q["test_cases"], mode="sample", enable_execution=True).status == "invalid"
    long_code = "x = 1\n" * 10_000
    assert evaluate_code(long_code, q["test_cases"], mode="sample", enable_execution=True).status == "invalid"


def test_invalid_mode_rejected(bank):
    with pytest.raises(ValueError):
        select_tests(bank["ARR-E-001"]["test_cases"], "quiz")