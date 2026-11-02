"""SQLAlchemy table definitions for SkillBridge AI."""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DIFFICULTIES = ("easy", "medium", "hard")
ASSESSMENT_MODES = ("sample", "judge", "quiz")
DATASET_TYPES = ("synthetic", "real")


def utc_now() -> datetime:
    """Current UTC time as a naive datetime (SQLite has no timezone type)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _in_list(column: str, values: tuple) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    assessments: Mapped[list["Assessment"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Question(Base):
    """Loaded from data/questions/*.json (the JSON files stay the source of truth)."""

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g. ARR-E-001
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(String(40), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(10), nullable=False)
    input_format: Mapped[str | None] = mapped_column(Text)
    output_format: Mapped[str | None] = mapped_column(Text)
    constraints: Mapped[str | None] = mapped_column(Text)
    examples: Mapped[list] = mapped_column(JSON, default=list)
    test_cases: Mapped[list] = mapped_column(JSON, default=list)
    learning_objectives: Mapped[list] = mapped_column(JSON, default=list)
    tags: Mapped[list] = mapped_column(JSON, default=list)

    __table_args__ = (
        CheckConstraint(_in_list("difficulty", DIFFICULTIES), name="ck_questions_difficulty"),
        Index("ix_questions_topic_difficulty", "topic", "difficulty"),
    )


class Assessment(Base):
    """One assessment session, which groups several attempts."""

    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    topic_filter: Mapped[str | None] = mapped_column(String(40))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    score: Mapped[float | None] = mapped_column(Float)

    user: Mapped["User"] = relationship(back_populates="assessments")
    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(_in_list("mode", ASSESSMENT_MODES), name="ck_assessments_mode"),
    )


class Submission(Base):
    """One attempt at one question."""

    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(
        ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # user_id is stored here too so per-user queries need no join.
    # The service layer always copies it from the assessment's owner.
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    code: Mapped[str | None] = mapped_column(Text)
    passed_tests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    exec_time_ms: Mapped[float | None] = mapped_column(Float)  # only if actually measured
    predicted_prob: Mapped[float | None] = mapped_column(Float)  # ML estimate BEFORE the attempt
    model_version: Mapped[str | None] = mapped_column(String(64))
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    assessment: Mapped["Assessment"] = relationship(back_populates="submissions")

    __table_args__ = (
        UniqueConstraint(
            "assessment_id", "question_id", "attempt_number", name="uq_submission_attempt"
        ),
        CheckConstraint("attempt_number >= 1", name="ck_submissions_attempt"),
        CheckConstraint(
            "passed_tests >= 0 AND passed_tests <= total_tests", name="ck_submissions_tests"
        ),
    )


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_prob: Mapped[float | None] = mapped_column(Float)
    strategy: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    was_attempted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(60), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(120), nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(10), nullable=False)
    n_train: Mapped[int] = mapped_column(Integer, nullable=False)
    n_test: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        CheckConstraint(_in_list("dataset_type", DATASET_TYPES), name="ck_model_dataset_type"),
    )


class SchemaVersion(Base):
    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)