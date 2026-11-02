"""Question bank: loading, validation, database sync and queries."""

import json
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import DIFFICULTIES, Question

TOPICS = (
    "Arrays", "Strings", "Hashing", "Searching", "Sorting", "Binary Search",
    "Two Pointers", "Sliding Window", "Recursion", "Linked Lists",
    "Stacks and Queues", "Trees", "Graphs", "Dynamic Programming",
)

_ID_RE = re.compile(r"^[A-Z]{2,5}-([EMH])-\d{3}$")
_DIFF_LETTER = {"easy": "E", "medium": "M", "hard": "H"}
_TEXT_FIELDS = ("title", "statement", "input_format", "output_format", "constraints")
_LIST_FIELDS = ("learning_objectives", "tags")


class QuestionBankError(Exception):
    """Raised when question files are missing or invalid."""


def validate_question(q: dict) -> None:
    """Raise QuestionBankError describing the first problem found."""
    if not isinstance(q, dict):
        raise QuestionBankError("Each question must be a JSON object.")
    qid = str(q.get("id", "<missing id>"))

    def fail(msg: str):
        raise QuestionBankError(f"{qid}: {msg}")

    match = _ID_RE.match(qid)
    if not match:
        fail("id must look like ARR-E-001 (PREFIX-E/M/H-number).")
    if q.get("difficulty") not in DIFFICULTIES:
        fail(f"difficulty must be one of {DIFFICULTIES}.")
    if match.group(1) != _DIFF_LETTER[q["difficulty"]]:
        fail("difficulty letter in the id does not match the difficulty field.")
    if q.get("topic") not in TOPICS:
        fail(f"topic must be one of {TOPICS}.")

    for field in _TEXT_FIELDS:
        if not isinstance(q.get(field), str) or not q[field].strip():
            fail(f"'{field}' must be a non-empty string.")
    for field in _LIST_FIELDS:
        value = q.get(field)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            fail(f"'{field}' must be a non-empty list of strings.")

    tests = q.get("test_cases")
    if not isinstance(tests, list) or len(tests) < 3:
        fail("'test_cases' must be a list with at least 3 entries.")
    for i, t in enumerate(tests):
        if not isinstance(t, dict) or not isinstance(t.get("input"), str) or not isinstance(t.get("output"), str):
            fail(f"test case {i} needs string 'input' and 'output'.")
        if "hidden" in t and not isinstance(t["hidden"], bool):
            fail(f"test case {i}: 'hidden' must be true or false.")
    if not any(not t.get("hidden", False) for t in tests):
        fail("at least one visible sample test case is required.")
    if not any(t.get("hidden", False) for t in tests):
        fail("at least one hidden test case is required.")

    if "reference_solution" in q:
        sol = q["reference_solution"]
        if not isinstance(sol, list) or not all(isinstance(line, str) for line in sol):
            fail("'reference_solution' must be a list of code lines.")


def load_question_bank(directory: Path) -> list[dict]:
    """Read and validate every *.json file in a directory."""
    directory = Path(directory)
    files = sorted(directory.glob("*.json"))
    if not files:
        raise QuestionBankError(f"No question files found in {directory}.")

    questions: list[dict] = []
    seen: set[str] = set()
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QuestionBankError(f"{path.name}: invalid JSON ({exc}).") from exc
        if not isinstance(data, list):
            raise QuestionBankError(f"{path.name}: top level must be a list of questions.")
        for q in data:
            validate_question(q)
            if q["id"] in seen:
                raise QuestionBankError(f"{q['id']}: duplicate question id.")
            seen.add(q["id"])
            questions.append(q)
    return questions


def _db_fields(q: dict) -> dict:
    tests = [
        {"input": t["input"], "output": t["output"], "hidden": bool(t.get("hidden", False))}
        for t in q["test_cases"]
    ]
    return {
        "title": q["title"].strip(),
        "statement": q["statement"].strip(),
        "topic": q["topic"],
        "difficulty": q["difficulty"],
        "input_format": q["input_format"],
        "output_format": q["output_format"],
        "constraints": q["constraints"],
        "examples": [t for t in tests if not t["hidden"]],
        "test_cases": tests,
        "learning_objectives": list(q["learning_objectives"]),
        "tags": list(q["tags"]),
    }  # reference_solution is deliberately NOT stored in the database


def sync_questions(session: Session, questions: list[dict]) -> dict:
    """Insert new questions and update changed ones. Never deletes (history stays valid)."""
    created = updated = unchanged = 0
    for q in questions:
        fields = _db_fields(q)
        row = session.get(Question, q["id"])
        if row is None:
            session.add(Question(id=q["id"], **fields))
            created += 1
            continue
        changed = False
        for key, value in fields.items():
            if getattr(row, key) != value:
                setattr(row, key, value)
                changed = True
        if changed:
            updated += 1
        else:
            unchanged += 1
    session.flush()
    return {"created": created, "updated": updated, "unchanged": unchanged}


def get_question(session: Session, question_id: str) -> Question | None:
    return session.get(Question, question_id)


def list_questions(
    session: Session, topic: str | None = None, difficulty: str | None = None
) -> list[Question]:
    stmt = select(Question).order_by(Question.id)
    if topic:
        stmt = stmt.where(Question.topic == topic)
    if difficulty:
        stmt = stmt.where(Question.difficulty == difficulty)
    return list(session.scalars(stmt))


def list_topics(session: Session) -> list[str]:
    """Topics present in the database, in the standard TOPICS order."""
    present = set(session.scalars(select(Question.topic).distinct()))
    return [t for t in TOPICS if t in present]


def get_sample_tests(question: Question) -> list[dict]:
    return [t for t in question.test_cases if not t.get("hidden", False)]


def public_view(question: Question) -> dict:
    """What the student may see: no hidden tests, no solutions."""
    return {
        "id": question.id,
        "title": question.title,
        "statement": question.statement,
        "topic": question.topic,
        "difficulty": question.difficulty,
        "input_format": question.input_format,
        "output_format": question.output_format,
        "constraints": question.constraints,
        "examples": get_sample_tests(question),
        "learning_objectives": question.learning_objectives,
        "tags": question.tags,
    }