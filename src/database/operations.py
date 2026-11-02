"""Database operations (queries and writes). Functions receive a Session."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import SchemaVersion, User

CURRENT_SCHEMA_VERSION = 1


def get_user_by_id(session: Session, user_id: int) -> User | None:
    return session.get(User, user_id)


def get_user_by_username(session: Session, username: str) -> User | None:
    return session.scalar(select(User).where(User.username == username))


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email))


def create_user(
    session: Session,
    *,
    username: str,
    email: str,
    password_hash: str,
    full_name: str | None = None,
) -> User:
    user = User(
        username=username, email=email, password_hash=password_hash, full_name=full_name
    )
    session.add(user)
    session.flush()  # assigns user.id and raises IntegrityError on duplicates
    return user


def count_users(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(User)) or 0


def get_schema_version(session: Session) -> int | None:
    return session.scalar(select(func.max(SchemaVersion.version)))


def ensure_schema_version(session: Session) -> int:
    """Record the current schema version once; do nothing if already recorded."""
    existing = get_schema_version(session)
    if existing is None:
        session.add(SchemaVersion(version=CURRENT_SCHEMA_VERSION))
        session.flush()
        return CURRENT_SCHEMA_VERSION
    return existing