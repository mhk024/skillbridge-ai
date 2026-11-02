"""Database operations (queries and writes). Functions receive a Session."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import (
    ASSESSMENT_MODES,
    Assessment,
    Question,
    SchemaVersion,
    Submission,
    User,
    utc_now,
)

CURRENT_SCHEMA_VERSION = 1


class AccessDeniedError(Exception):
    """The record does not exist for this user (deliberately indistinguishable)."""


# ----------------------------------------------------------------------- users
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


# ----------------------------------------------------------------- schema version
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


# ----------------------------------------------------------------- assessments
def create_assessment(
    session: Session, *, user_id: int, mode: str, topic_filter: str | None = None
) -> Assessment:
    if mode not in ASSESSMENT_MODES:
        raise ValueError(f"mode must be one of {ASSESSMENT_MODES}")
    assessment = Assessment(user_id=user_id, mode=mode, topic_filter=topic_filter)
    session.add(assessment)
    session.flush()
    return assessment


def get_assessment_for_user(
    session: Session, assessment_id: int, user_id: int
) -> Assessment | None:
    """Returns the assessment only if it belongs to this user."""
    return session.scalar(
        select(Assessment).where(
            Assessment.id == assessment_id, Assessment.user_id == user_id
        )
    )


def count_attempts(session: Session, user_id: int, question_id: str) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(Submission)
            .where(Submission.user_id == user_id, Submission.question_id == question_id)
        )
        or 0
    )


def record_submission(
    session: Session,
    *,
    user_id: int,
    assessment_id: int,
    question_id: str,
    code: str | None,
    passed_tests: int,
    total_tests: int,
    exec_time_ms: float | None = None,
    predicted_prob: float | None = None,
    model_version: str | None = None,
) -> Submission:
    """Save one attempt. attempt_number counts this user's earlier attempts on the question."""
    assessment = get_assessment_for_user(session, assessment_id, user_id)
    if assessment is None:
        raise AccessDeniedError("Assessment not found for this user.")
    if assessment.completed_at is not None:
        raise ValueError("This assessment is already completed.")
    if session.get(Question, question_id) is None:
        raise ValueError(f"Unknown question: {question_id}")
    if passed_tests < 0 or total_tests < 0 or passed_tests > total_tests:
        raise ValueError("Test counts are invalid.")

    submission = Submission(
        assessment_id=assessment_id,
        user_id=user_id,
        question_id=question_id,
        attempt_number=count_attempts(session, user_id, question_id) + 1,
        code=code,
        passed_tests=passed_tests,
        total_tests=total_tests,
        is_correct=total_tests > 0 and passed_tests == total_tests,
        exec_time_ms=exec_time_ms,
        predicted_prob=predicted_prob,
        model_version=model_version,
    )
    session.add(submission)
    session.flush()
    return submission


def complete_assessment(session: Session, assessment_id: int, user_id: int) -> Assessment:
    """Close an assessment. score = % of attempted questions solved at least once."""
    assessment = get_assessment_for_user(session, assessment_id, user_id)
    if assessment is None:
        raise AccessDeniedError("Assessment not found for this user.")
    if assessment.completed_at is None:
        attempted = {s.question_id for s in assessment.submissions}
        solved = {s.question_id for s in assessment.submissions if s.is_correct}
        assessment.score = round(100 * len(solved) / len(attempted), 1) if attempted else None
        assessment.completed_at = utc_now()
        session.flush()
    return assessment


def get_user_assessments(session: Session, user_id: int) -> list[Assessment]:
    return list(
        session.scalars(
            select(Assessment)
            .where(Assessment.user_id == user_id)
            .order_by(Assessment.started_at.desc(), Assessment.id.desc())
        )
    )


def get_user_submissions(
    session: Session, user_id: int, limit: int | None = None
) -> list[Submission]:
    stmt = (
        select(Submission)
        .where(Submission.user_id == user_id)
        .order_by(Submission.submitted_at.desc(), Submission.id.desc())
    )
    if limit:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))



# ------------------------------------------------- assessment helpers (Phase 4B)
def get_open_assessment(session: Session, user_id: int) -> Assessment | None:
    """The user's most recent unfinished assessment, if any."""
    return session.scalar(
        select(Assessment)
        .where(Assessment.user_id == user_id, Assessment.completed_at.is_(None))
        .order_by(Assessment.started_at.desc(), Assessment.id.desc())
        .limit(1)
    )


def get_attempted_question_ids(session: Session, user_id: int) -> set[str]:
    return set(
        session.scalars(
            select(Submission.question_id).where(Submission.user_id == user_id).distinct()
        )
    )


def get_solved_question_ids(session: Session, user_id: int) -> set[str]:
    return set(
        session.scalars(
            select(Submission.question_id)
            .where(Submission.user_id == user_id, Submission.is_correct.is_(True))
            .distinct()
        )
    )


def get_question_submissions(
    session: Session, user_id: int, question_id: str
) -> list[Submission]:
    """This user's attempts on one question, newest first."""
    return list(
        session.scalars(
            select(Submission)
            .where(Submission.user_id == user_id, Submission.question_id == question_id)
            .order_by(Submission.submitted_at.desc(), Submission.id.desc())
        )
    )


def get_assessment_progress(session: Session, assessment_id: int, user_id: int) -> dict:
    assessment = get_assessment_for_user(session, assessment_id, user_id)
    if assessment is None:
        raise AccessDeniedError("Assessment not found for this user.")
    attempted = {s.question_id for s in assessment.submissions}
    solved = {s.question_id for s in assessment.submissions if s.is_correct}
    return {
        "attempts": len(assessment.submissions),
        "attempted": len(attempted),
        "solved": len(solved),
    }