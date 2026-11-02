import pytest

from src.auth import service as auth_service
from src.database.connection import create_db_engine, init_db


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