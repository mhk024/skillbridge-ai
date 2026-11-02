import pytest

from src.assessment import question_service as qs
from src.assessment.evaluation_service import evaluate_code
from src.assessment.submission_service import record_evaluation
from src.database import operations as ops
from src.database.connection import session_scope
from src.utils.config import get_settings

Q1, Q2 = "ARR-E-001", "STR-E-001"


@pytest.fixture(scope="module")
def bank():
    return {q["id"]: q for q in qs.load_question_bank(get_settings().questions_dir)}


def make_user(engine, name):
    with session_scope(engine) as s:
        return ops.create_user(s, username=name, email=f"{name}@example.com", password_hash="x" * 60).id


def start(engine, uid):
    with session_scope(engine) as s:
        return ops.create_assessment(s, user_id=uid, mode="judge").id


def add(engine, uid, aid, qid, passed, total):
    with session_scope(engine) as s:
        ops.record_submission(
            s, user_id=uid, assessment_id=aid, question_id=qid,
            code="x", passed_tests=passed, total_tests=total,
        )


def test_every_topic_has_easy_and_medium_questions(bank):
    for topic in qs.TOPICS:
        difficulties = {q["difficulty"] for q in bank.values() if q["topic"] == topic}
        assert {"easy", "medium"} <= difficulties, f"{topic} is missing easy or medium"


def test_open_assessment_found_then_closed(seeded_engine):
    uid = make_user(seeded_engine, "alice")
    with session_scope(seeded_engine) as s:
        assert ops.get_open_assessment(s, uid) is None
    aid = start(seeded_engine, uid)
    with session_scope(seeded_engine) as s:
        assert ops.get_open_assessment(s, uid).id == aid
        ops.complete_assessment(s, aid, uid)
    with session_scope(seeded_engine) as s:
        assert ops.get_open_assessment(s, uid) is None


def test_open_assessment_is_per_user(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    start(seeded_engine, alice)
    with session_scope(seeded_engine) as s:
        assert ops.get_open_assessment(s, bob) is None


def test_attempted_and_solved_ids(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    aid = start(seeded_engine, alice)
    add(seeded_engine, alice, aid, Q1, 5, 5)
    add(seeded_engine, alice, aid, Q2, 1, 5)
    with session_scope(seeded_engine) as s:
        assert ops.get_attempted_question_ids(s, alice) == {Q1, Q2}
        assert ops.get_solved_question_ids(s, alice) == {Q1}
        assert ops.get_attempted_question_ids(s, bob) == set()


def test_question_history_newest_first_and_isolated(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    aid = start(seeded_engine, alice)
    add(seeded_engine, alice, aid, Q1, 1, 5)
    add(seeded_engine, alice, aid, Q1, 5, 5)
    add(seeded_engine, bob, start(seeded_engine, bob), Q1, 0, 5)
    with session_scope(seeded_engine) as s:
        mine = ops.get_question_submissions(s, alice, Q1)
        assert [x.attempt_number for x in mine] == [2, 1]
        assert all(x.user_id == alice for x in mine)


def test_assessment_progress_and_ownership(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    aid = start(seeded_engine, alice)
    add(seeded_engine, alice, aid, Q1, 1, 5)
    add(seeded_engine, alice, aid, Q1, 5, 5)
    add(seeded_engine, alice, aid, Q2, 0, 5)
    with session_scope(seeded_engine) as s:
        assert ops.get_assessment_progress(s, aid, alice) == {"attempts": 3, "attempted": 2, "solved": 1}
        with pytest.raises(ops.AccessDeniedError):
            ops.get_assessment_progress(s, aid, bob)


def test_correct_judged_run_is_saved(seeded_engine, bank):
    q = bank[Q1]
    code = "\n".join(q["reference_solution"]) + "\n"
    ev = evaluate_code(code, q["test_cases"], mode="judge", enable_execution=True)
    uid = make_user(seeded_engine, "alice")
    aid = start(seeded_engine, uid)
    with session_scope(seeded_engine) as s:
        sub = record_evaluation(
            s, user_id=uid, assessment_id=aid, question_id=Q1, code=code, evaluation=ev
        )
        assert sub.is_correct and sub.attempt_number == 1
        assert sub.passed_tests == sub.total_tests == len(q["test_cases"])
        assert sub.exec_time_ms is not None


def test_syntax_error_saved_as_failed_attempt(seeded_engine, bank):
    q = bank[Q1]
    ev = evaluate_code("def f(:\n", q["test_cases"], mode="judge", enable_execution=True)
    uid = make_user(seeded_engine, "alice")
    aid = start(seeded_engine, uid)
    with session_scope(seeded_engine) as s:
        sub = record_evaluation(
            s, user_id=uid, assessment_id=aid, question_id=Q1, code="def f(:", evaluation=ev
        )
        assert not sub.is_correct and sub.passed_tests == 0
        assert sub.total_tests == len(q["test_cases"])
        assert sub.exec_time_ms is None


def test_sample_disabled_and_invalid_runs_are_not_saved(seeded_engine, bank):
    q = bank[Q1]
    uid = make_user(seeded_engine, "alice")
    aid = start(seeded_engine, uid)
    sample = evaluate_code("print(1)", q["test_cases"], mode="sample", enable_execution=True)
    disabled = evaluate_code("print(1)", q["test_cases"], mode="judge", enable_execution=False)
    empty = evaluate_code("  ", q["test_cases"], mode="judge", enable_execution=True)
    for ev in (sample, disabled, empty):
        with pytest.raises(ValueError):
            with session_scope(seeded_engine) as s:
                record_evaluation(
                    s, user_id=uid, assessment_id=aid, question_id=Q1, code="x", evaluation=ev
                )
    with session_scope(seeded_engine) as s:
        assert ops.get_user_submissions(s, uid) == []