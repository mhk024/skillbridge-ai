import pandas as pd
import pytest

from src.assessment.question_service import TOPICS, load_question_bank
from src.ml import predict as pr
from src.ml import train as tr
from src.ml.generate_synthetic import generate_attempts
from src.ml.recommend import PREREQUISITES, make_predictor, recommend, topic_statuses
from src.utils.config import get_settings

ARR_E = ("ARR-E-001", "Arrays", "easy")
ARR_M = ("ARR-M-001", "Arrays", "medium")
SORT_E = ("SORT-E-001", "Sorting", "easy")


@pytest.fixture(scope="module")
def bank():
    return [
        {"id": q["id"], "title": q["title"], "topic": q["topic"], "difficulty": q["difficulty"]}
        for q in load_question_bank(get_settings().questions_dir)
    ]


@pytest.fixture(scope="module")
def model():
    attempts = generate_attempts(n_students=60, seed=5)
    artifact, _ = tr.train_and_evaluate(
        attempts, dataset_name="test_synthetic", dataset_type="synthetic", seed=5, rf_trees=20
    )
    return pr.model_from_artifact(artifact)


def history(rows):
    """rows: (question_id, topic, difficulty, is_correct) for one student, oldest first."""
    start = pd.Timestamp("2026-01-01 09:00:00")
    return pd.DataFrame(
        [
            {
                "attempt_id": i + 1, "user_id": 1, "question_id": qid, "topic": topic,
                "difficulty": diff, "attempted_at": start + pd.Timedelta(minutes=i),
                "is_correct": outcome,
            }
            for i, (qid, topic, diff, outcome) in enumerate(rows)
        ]
    )


def q(qid, topic, difficulty):
    return {"id": qid, "title": qid, "topic": topic, "difficulty": difficulty}


CUSTOM = [
    q("T-E-1", "Arrays", "easy"), q("T-E-2", "Arrays", "easy"),
    q("T-M-1", "Arrays", "medium"), q("T-H-1", "Arrays", "hard"),
]


def only_arrays(recs):
    return next(r for r in recs if r.topic == "Arrays")


# ------------------------------------------------------------------- configuration
def test_prerequisite_map_uses_real_topics():
    for topic, prereqs in PREREQUISITES.items():
        assert topic in TOPICS and topic not in prereqs
        assert all(p in TOPICS for p in prereqs)


# ------------------------------------------------------------------- topic status
@pytest.mark.parametrize(
    "rows, state",
    [
        ([], "new"),
        ([(*ARR_E, 0)] * 3, "struggling"),
        ([(*ARR_E, 1)] * 3, "strong"),
        ([(*ARR_E, 1), (*ARR_E, 0)], "developing"),  # fewer than 3 attempts: no verdict yet
    ],
)
def test_topic_state_classification(rows, state):
    assert topic_statuses(history(rows) if rows else None)["Arrays"].state == state


def test_status_numbers_match_hand_calculation():
    s = topic_statuses(history([(*ARR_E, 1), (*ARR_E, 0), (*ARR_M, 1)]))["Arrays"]
    assert s.attempts == 3
    assert s.accuracy == pytest.approx(0.6)  # (2 + 1) / (3 + 2)
    assert s.solved == {1: 1, 2: 1, 3: 0} and s.highest_solved == 2
    assert s.state == "developing"


def test_accuracy_follows_the_recent_window():
    s = topic_statuses(history([(*ARR_E, 0)] * 5 + [(*ARR_E, 1)] * 5))["Arrays"]
    assert s.attempts == 10
    assert s.accuracy == pytest.approx(0.6)  # last 8 attempts: 5 correct -> (5 + 1) / (8 + 2)


# ----------------------------------------------------------- the rules (no model)
def test_new_student_gets_easy_basics_in_foundational_order(bank):
    recs = recommend(None, bank, n=5)
    assert [r.topic for r in recs] == ["Arrays", "Strings", "Hashing", "Searching", "Sorting"]
    assert all(r.difficulty == "easy" and r.strategy == "explore" for r in recs)
    assert all(r.predicted_prob is None and r.reason for r in recs)


def test_struggling_topic_comes_first_with_easy_foundation(bank):
    recs = recommend(history([(*ARR_E, 0)] * 5), bank, n=14)
    assert recs[0].topic == "Arrays"
    assert (recs[0].difficulty, recs[0].strategy) == ("easy", "foundation")
    assert "attempted it before" in recs[0].reason


def test_strong_topic_steps_up_a_level(bank):
    recs = recommend(history([(*ARR_E, 1)] * 3), bank, n=14)
    arrays = only_arrays(recs)
    assert (arrays.question_id, arrays.strategy) == ("ARR-M-001", "step_up")
    assert recs[0].topic != "Arrays"  # a strong topic is a lower priority than untouched ones


def test_recommendations_change_when_performance_changes(bank):
    failing = recommend(history([(*ARR_E, 0)] * 5), bank, n=14)
    succeeding = recommend(history([(*ARR_E, 1)] * 3), bank, n=14)
    assert (only_arrays(failing).difficulty, only_arrays(failing).strategy) == ("easy", "foundation")
    assert (only_arrays(succeeding).difficulty, only_arrays(succeeding).strategy) == ("medium", "step_up")
    assert [r.question_id for r in failing] != [r.question_id for r in succeeding]


def test_recovery_changes_recommendation_back(bank):
    struggling = recommend(history([(*ARR_E, 0)] * 4), bank, n=14)
    recovered = recommend(history([(*ARR_E, 0)] * 4 + [(*ARR_E, 1)] * 4), bank, n=14)
    assert only_arrays(struggling).strategy == "foundation"
    assert only_arrays(recovered).strategy != "foundation"  # recent success lifts it out of "struggling"


def test_weak_prerequisite_is_practised_before_its_dependent_topic(bank):
    rows = [(*ARR_E, 0)] * 4 + [(*SORT_E, 0)]
    recs = recommend(history(rows), bank, n=14)
    assert recs[0].topic == "Arrays"
    assert "foundation for" in recs[0].reason and "Sorting" in recs[0].reason
    order = [r.topic for r in recs]
    assert order.index("Arrays") < order.index("Sorting")


def test_solved_and_excluded_questions_are_never_offered(bank):
    recs = recommend(history([(*ARR_E, 1)]), bank, excluded_ids={"STR-E-001"}, n=14)
    ids = {r.question_id for r in recs}
    assert "ARR-E-001" not in ids and "STR-E-001" not in ids


def test_no_duplicates_and_respects_n(bank):
    recs = recommend(None, bank, n=20)
    ids = [r.question_id for r in recs]
    assert len(ids) == len(set(ids)) and len(ids) <= 20
    assert len(recommend(None, bank, n=3)) == 3


def test_fully_solved_topic_gives_no_recommendation():
    questions = [q("A-1", "Arrays", "easy"), q("S-1", "Strings", "easy")]
    recs = recommend(history([("A-1", "Arrays", "easy", 1)]), questions)
    assert [r.question_id for r in recs] == ["S-1"]
    assert recommend(history([("A-1", "Arrays", "easy", 1), ("S-1", "Strings", "easy", 1)]), questions) == []


def test_invalid_inputs_rejected(bank):
    with pytest.raises(ValueError):
        recommend(None, bank, n=0)
    with pytest.raises(ValueError):
        recommend(None, [q("X", "Astrology", "easy")])
    with pytest.raises(ValueError):
        recommend(None, [q("X", "Arrays", "impossible")])


# --------------------------------------------------------- the ML supports the rules
def test_without_model_strong_topic_steps_up():
    rec = recommend(history([("T-E-1", "Arrays", "easy", 1)] * 3), CUSTOM, n=1)[0]
    assert (rec.question_id, rec.strategy) == ("T-M-1", "step_up")


def test_low_model_estimate_nudges_one_level_down():
    rec = recommend(
        history([("T-E-1", "Arrays", "easy", 1)] * 3), CUSTOM, predict=lambda x: 0.1, n=1
    )[0]
    assert (rec.question_id, rec.strategy) == ("T-E-2", "ml_step_down")
    assert rec.predicted_prob == pytest.approx(0.1) and "model" in rec.reason.lower()


def test_high_model_estimate_nudges_one_level_up_but_not_when_struggling():
    developing = history([("T-E-1", "Arrays", "easy", 1), ("T-E-2", "Arrays", "easy", 0), ("T-E-2", "Arrays", "easy", 0)])
    up = recommend(developing, CUSTOM, predict=lambda x: 0.9, n=1)[0]
    assert (up.question_id, up.strategy) == ("T-M-1", "ml_step_up")

    struggling = history([("T-E-2", "Arrays", "easy", 0)] * 4)
    held = recommend(struggling, CUSTOM, predict=lambda x: 0.95, n=1)[0]
    assert held.strategy == "foundation" and held.difficulty == "easy"


def test_model_picks_the_question_inside_the_productive_zone():
    estimates = {"T-E-1": 0.95, "T-E-2": 0.6}
    rec = recommend(None, CUSTOM, predict=lambda x: estimates.get(x["id"], 0.6), n=1)[0]
    assert rec.question_id == "T-E-2"


def test_model_failure_does_not_break_recommendations():
    def boom(question):
        raise RuntimeError("model exploded")

    recs = recommend(None, CUSTOM, predict=boom, n=1)
    assert len(recs) == 1 and recs[0].predicted_prob is None


def test_real_model_estimates_are_attached_and_stated(bank, model):
    hist = history([(*ARR_E, 0)] * 4)
    recs = recommend(hist, bank, predict=make_predictor(model, hist, 1), n=5)
    assert recs
    assert all(r.predicted_prob is not None and 0.0 <= r.predicted_prob <= 1.0 for r in recs)
    assert all("Model estimate" in r.reason for r in recs)
    new_student = recommend(None, bank, predict=make_predictor(model, None, 1), n=3)
    assert len(new_student) == 3 and all(r.predicted_prob is not None for r in new_student)