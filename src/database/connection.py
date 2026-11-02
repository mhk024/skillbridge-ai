"""Database engine, sessions and initialization."""

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from src.database import operations
from src.database.models import Base
from src.utils.config import get_settings


def create_db_engine(database_url: str) -> Engine:
    """Create an engine. For SQLite: create the folder and enforce foreign keys."""
    is_sqlite = database_url.startswith("sqlite")
    if is_sqlite:
        db_path = make_url(database_url).database
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(database_url, connect_args=connect_args)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


@lru_cache(maxsize=None)
def get_engine() -> Engine:
    """Shared engine for the running app (built from settings)."""
    return create_db_engine(get_settings().database_url)


@contextmanager
def session_scope(engine: Engine | None = None):
    """Transaction scope: commit if the block succeeds, otherwise roll back."""
    engine = engine or get_engine()
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    """Create missing tables and record the schema version. Safe to run repeatedly."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    with session_scope(engine) as session:
        operations.ensure_schema_version(session)