import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from src.database import operations as ops
from src.database.connection import create_db_engine, init_db, session_scope
from src.database.models import Assessment, Question

EXPECTED_TABLES = {
    "users", "questions", "assessments", "submissions",
    "recommendations", "model_metadata", "schema_version",
}


def test_all_tables_created(engine):
    assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())


def test_init_db_is_idempotent_and_records_one_version(engine):
    init_db(engine)
    init_db(engine)
    with session_scope(engine) as s:
        assert ops.get_schema_version(s) == ops.CURRENT_SCHEMA_VERSION


def test_data_persists_after_engine_restart(db_url):
    first = create_db_engine(db_url)
    init_db(first)
    with session_scope(first) as s:
        ops.create_user(s, username="carol", email="c@example.com", password_hash="x" * 60)
    first.dispose()

    second = create_db_engine(db_url)  # simulates restarting the app
    with session_scope(second) as s:
        assert ops.get_user_by_username(s, "carol") is not None
        assert ops.count_users(s) == 1
    second.dispose()


def test_failed_transaction_rolls_back(engine):
    with pytest.raises(RuntimeError):
        with session_scope(engine) as s:
            ops.create_user(s, username="dave", email="d@example.com", password_hash="x" * 60)
            raise RuntimeError("boom")
    with session_scope(engine) as s:
        assert ops.get_user_by_username(s, "dave") is None


def test_database_enforces_unique_username(engine):
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            ops.create_user(s, username="eve", email="e1@example.com", password_hash="x")
            ops.create_user(s, username="eve", email="e2@example.com", password_hash="x")


def test_foreign_keys_are_enforced(engine):
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            s.add(Assessment(user_id=9999, mode="quiz"))
            s.flush()


def test_invalid_difficulty_rejected(engine):
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            s.add(Question(id="X-1", title="t", statement="s", topic="arrays", difficulty="impossible"))
            s.flush()