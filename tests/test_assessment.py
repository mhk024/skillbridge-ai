import pytest

from src.assessment.evaluation_service import evaluate_code
from src.assessment.question_service import load_question_bank
from src.database import operations as ops
from src.database.connection import create_db_engine, session_scope
from src.utils.config import get_settings

Q1, Q2 = "ARR-E-001", "STR-E-001"


def make_user(engine, name):
    with session_scope(engine) as s:
        return ops.create_user(s, username=name, email=f"{name}@example.com", password_hash="x" * 60).id


def start(engine, user_id, mode="judge"):
    with session_scope(engine) as s:
        return ops.create_assessment(s, user_id=user_id, mode=mode).id


def submit(engine, user_id, assessment_id, qid, passed, total):
    with session_scope(engine) as s:
        sub = ops.record_submission(
            s, user_id=user_id, assessment_id=assessment_id, question_id=qid,
            code="print(1)", passed_tests=passed, total_tests=total,
        )
        return sub.id


def test_record_submission_sets_correctness(seeded_engine):
    uid = make_user(seeded_engine, "alice")
    aid = start(seeded_engine, uid)
    submit(seeded_engine, uid, aid, Q1, 5, 5)
    submit(seeded_engine, uid, aid, Q2, 2, 5)
    with session_scope(seeded_engine) as s:
        subs = {x.question_id: x for x in ops.get_user_submissions(s, uid)}
        assert subs[Q1].is_correct is True
        assert subs[Q2].is_correct is False
        assert subs[Q1].user_id == uid


def test_attempt_numbers_increase_across_assessments(seeded_engine):
    uid = make_user(seeded_engine, "alice")
    a1, a2 = start(seeded_engine, uid), start(seeded_engine, uid)
    submit(seeded_engine, uid, a1, Q1, 1, 5)
    submit(seeded_engine, uid, a1, Q1, 2, 5)
    submit(seeded_engine, uid, a2, Q1, 5, 5)
    with session_scope(seeded_engine) as s:
        numbers = sorted(x.attempt_number for x in ops.get_user_submissions(s, uid))
        assert numbers == [1, 2, 3]


def test_cannot_submit_into_another_users_assessment(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    alice_assessment = start(seeded_engine, alice)
    with pytest.raises(ops.AccessDeniedError):
        submit(seeded_engine, bob, alice_assessment, Q1, 5, 5)
    with session_scope(seeded_engine) as s:
        assert ops.get_user_submissions(s, alice) == []


def test_get_assessment_for_wrong_owner_returns_none(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    aid = start(seeded_engine, alice)
    with session_scope(seeded_engine) as s:
        assert ops.get_assessment_for_user(s, aid, bob) is None
        assert ops.get_assessment_for_user(s, aid, alice) is not None


def test_queries_are_isolated_per_user(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    submit(seeded_engine, alice, start(seeded_engine, alice), Q1, 5, 5)
    submit(seeded_engine, bob, start(seeded_engine, bob), Q2, 0, 5)
    with session_scope(seeded_engine) as s:
        assert {x.question_id for x in ops.get_user_submissions(s, alice)} == {Q1}
        assert {x.question_id for x in ops.get_user_submissions(s, bob)} == {Q2}
        assert len(ops.get_user_assessments(s, alice)) == 1


def test_unknown_question_and_bad_counts_rejected(seeded_engine):
    uid = make_user(seeded_engine, "alice")
    aid = start(seeded_engine, uid)
    with pytest.raises(ValueError):
        submit(seeded_engine, uid, aid, "NOPE-E-999", 1, 1)
    with pytest.raises(ValueError):
        submit(seeded_engine, uid, aid, Q1, 6, 5)
    with pytest.raises(ValueError):
        submit(seeded_engine, uid, aid, Q1, -1, 5)


def test_invalid_assessment_mode_rejected(seeded_engine):
    uid = make_user(seeded_engine, "alice")
    with pytest.raises(ValueError):
        start(seeded_engine, uid, mode="cheating")


def test_complete_assessment_score_and_lock(seeded_engine):
    uid = make_user(seeded_engine, "alice")
    aid = start(seeded_engine, uid)
    submit(seeded_engine, uid, aid, Q1, 1, 5)
    submit(seeded_engine, uid, aid, Q1, 5, 5)  # solved on the second attempt
    submit(seeded_engine, uid, aid, Q2, 0, 5)  # never solved
    with session_scope(seeded_engine) as s:
        a = ops.complete_assessment(s, aid, uid)
        assert a.score == 50.0 and a.completed_at is not None
    with pytest.raises(ValueError):
        submit(seeded_engine, uid, aid, Q1, 5, 5)


def test_complete_assessment_requires_ownership(seeded_engine):
    alice, bob = make_user(seeded_engine, "alice"), make_user(seeded_engine, "bob")
    aid = start(seeded_engine, alice)
    with pytest.raises(ops.AccessDeniedError):
        with session_scope(seeded_engine) as s:
            ops.complete_assessment(s, aid, bob)


def test_submissions_persist_after_restart(seeded_engine, db_url):
    uid = make_user(seeded_engine, "alice")
    submit(seeded_engine, uid, start(seeded_engine, uid), Q1, 5, 5)
    seeded_engine.dispose()
    restarted = create_db_engine(db_url)
    with session_scope(restarted) as s:
        assert len(ops.get_user_submissions(s, uid)) == 1
    restarted.dispose()


def test_evaluate_then_record_end_to_end(seeded_engine):
    bank = {q["id"]: q for q in load_question_bank(get_settings().questions_dir)}
    q = bank[Q1]
    ev = evaluate_code(
        "\n".join(q["reference_solution"]) + "\n", q["test_cases"], mode="judge", enable_execution=True
    )
    uid = make_user(seeded_engine, "alice")
    aid = start(seeded_engine, uid)
    with session_scope(seeded_engine) as s:
        sub = ops.record_submission(
            s, user_id=uid, assessment_id=aid, question_id=Q1, code="...",
            passed_tests=ev.passed, total_tests=ev.total, exec_time_ms=ev.avg_duration_ms,
        )
        assert sub.is_correct is True
        assert sub.exec_time_ms is not None