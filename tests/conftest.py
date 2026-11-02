import pytest

from src.assessment.question_service import load_question_bank, sync_questions
from src.auth import service as auth_service
from src.database.connection import create_db_engine, init_db, session_scope
from src.utils.config import get_settings


@pytest.fixture(autouse=True)
def fast_bcrypt(monkeypatch):
    """Use the minimum bcrypt cost in tests so they run quickly."""
    monkeypatch.setattr(auth_service, "BCRYPT_ROUNDS", 4)


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{(tmp_path / 'test.db').as_posix()}"


@pytest.fixture
def engine(db_url):
    engine = create_db_engine(db_url)
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def seeded_engine(engine):
    """Database with the real question bank loaded."""
    questions = load_question_bank(get_settings().questions_dir)
    with session_scope(engine) as session:
        sync_questions(session, questions)
    return engine