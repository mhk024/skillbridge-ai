import copy

import pytest

from scripts.validate_questions import validate_all
from src.assessment import question_service as qs
from src.database.connection import session_scope
from src.database.models import Question
from src.utils.config import get_settings


@pytest.fixture(scope="module")
def bank():
    return qs.load_question_bank(get_settings().questions_dir)


@pytest.fixture
def valid_question():
    return {
        "id": "ARR-E-999", "title": "T", "topic": "Arrays", "difficulty": "easy",
        "statement": "S", "input_format": "I", "output_format": "O", "constraints": "C",
        "learning_objectives": ["learn"], "tags": ["tag"],
        "test_cases": [
            {"input": "1\n", "output": "1", "hidden": False},
            {"input": "2\n", "output": "2", "hidden": True},
            {"input": "3\n", "output": "3", "hidden": True},
        ],
    }


def test_bank_loads_and_ids_unique(bank):
    ids = [q["id"] for q in bank]
    assert len(ids) == len(set(ids)) >= 15


def test_bank_covers_all_topics_and_difficulties(bank):
    assert {q["topic"] for q in bank} == set(qs.TOPICS)
    assert {q["difficulty"] for q in bank} == {"easy", "medium", "hard"}


def test_reference_solutions_pass_their_own_tests(bank):
    failures = {qid: p for qid, p in validate_all(bank).items() if p}
    assert failures == {}


def test_valid_question_accepted(valid_question):
    qs.validate_question(valid_question)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda q: q.pop("title"),
        lambda q: q.update(difficulty="impossible"),
        lambda q: q.update(id="ARR-M-999"),  # letter does not match 'easy'
        lambda q: q.update(id="bad id"),
        lambda q: q.update(topic="Astrology"),
        lambda q: q.update(test_cases=q["test_cases"][:2]),
        lambda q: [t.update(hidden=False) for t in q["test_cases"]],  # no hidden tests
        lambda q: [t.update(hidden=True) for t in q["test_cases"]],   # no sample
        lambda q: q.update(tags=[]),
    ],
)
def test_invalid_questions_rejected(valid_question, mutate):
    q = copy.deepcopy(valid_question)
    mutate(q)
    with pytest.raises(qs.QuestionBankError):
        qs.validate_question(q)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(qs.QuestionBankError):
        qs.load_question_bank(tmp_path)


def test_sync_is_idempotent_and_updates(engine, bank):
    with session_scope(engine) as s:
        first = qs.sync_questions(s, bank)
    assert first["created"] == len(bank)
    with session_scope(engine) as s:
        second = qs.sync_questions(s, bank)
    assert second == {"created": 0, "updated": 0, "unchanged": len(bank)}

    changed = copy.deepcopy(bank)
    changed[0]["title"] = "Renamed title"
    with session_scope(engine) as s:
        assert qs.sync_questions(s, changed)["updated"] == 1
        assert s.get(Question, changed[0]["id"]).title == "Renamed title"


def test_reference_solution_never_stored(engine, bank):
    with session_scope(engine) as s:
        qs.sync_questions(s, bank)
        row = qs.get_question(s, "ARR-E-001")
        assert not hasattr(row, "reference_solution")


def test_public_view_hides_hidden_tests(engine, bank):
    with session_scope(engine) as s:
        qs.sync_questions(s, bank)
        q = qs.get_question(s, "ARR-E-001")
        view = qs.public_view(q)
        assert all(not t["hidden"] for t in view["examples"])
        assert "test_cases" not in view
        assert len(q.test_cases) > len(view["examples"])


def test_filtering_and_topics(engine, bank):
    with session_scope(engine) as s:
        qs.sync_questions(s, bank)
        assert {q.topic for q in qs.list_questions(s, topic="Arrays")} == {"Arrays"}
        assert {q.difficulty for q in qs.list_questions(s, difficulty="hard")} == {"hard"}
        assert qs.list_topics(s) == list(qs.TOPICS)