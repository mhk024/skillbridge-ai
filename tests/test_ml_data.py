import numpy as np
import pandas as pd
import pytest

from src.assessment.question_service import TOPICS
from src.ml import preprocessing as pp
from src.ml.generate_synthetic import generate_attempts, save_dataset


@pytest.fixture(scope="module")
def synthetic():
    return generate_attempts(n_students=150, seed=7)


@pytest.fixture(scope="module")
def features(synthetic):
    return pp.build_features(synthetic)


def attempts_frame(rows):
    """rows: (question_id, topic, difficulty, is_correct) for ONE user, in time order."""
    base = pd.Timestamp("2025-03-01 10:00:00")
    return pd.DataFrame(
        [
            {
                "attempt_id": i + 1, "user_id": 1, "question_id": q, "topic": t,
                "difficulty": d, "attempted_at": base + pd.Timedelta(minutes=i), "is_correct": c,
            }
            for i, (q, t, d, c) in enumerate(rows)
        ]
    )


HAND_ROWS = [
    ("Q1", "Arrays", "easy", 0),
    ("Q1", "Arrays", "easy", 1),
    ("Q2", "Strings", "easy", 1),
    ("Q3", "Arrays", "medium", 0),
    ("Q1", "Arrays", "easy", 1),
    ("Q4", "Arrays", "easy", 0),
]


# ------------------------------------------------------------------ the generator
def test_generator_is_deterministic_for_same_seed():
    pd.testing.assert_frame_equal(generate_attempts(20, 1), generate_attempts(20, 1))


def test_different_seeds_give_different_data():
    assert not generate_attempts(20, 1)["is_correct"].equals(generate_attempts(20, 2)["is_correct"])


def test_generated_table_is_valid_and_diagnostic_prob_in_range(synthetic):
    pp.validate_attempts(synthetic)
    assert synthetic["sim_true_prob"].between(0, 1).all()
    assert set(synthetic["topic"]) <= set(TOPICS)


def test_times_increase_within_each_user(synthetic):
    for _, g in synthetic.groupby("user_id"):
        assert g["attempted_at"].is_monotonic_increasing


def test_harder_questions_are_solved_less_often(synthetic):
    rate = synthetic.groupby("difficulty")["is_correct"].mean()
    assert rate["easy"] > rate["medium"] > rate["hard"]


def test_dataset_saved_and_reloadable(synthetic, tmp_path):
    path = save_dataset(synthetic, tmp_path, seed=7)
    assert (tmp_path / "synthetic_attempts.meta.json").exists()
    reloaded = pd.read_csv(path)
    assert pp.build_features(reloaded).shape == pp.build_features(synthetic).shape


# --------------------------------------------------------------- hand-checked values
def test_features_match_hand_calculation():
    f = pp.build_features(attempts_frame(HAND_ROWS)).set_index("attempt_id")

    first = f.loc[1]
    assert first["user_prior_attempts"] == 0
    assert first["user_prior_accuracy"] == pytest.approx(0.5)
    assert first["topic_prior_accuracy"] == pytest.approx(0.5)

    second = f.loc[2]  # retry of Q1 after one failure
    assert second["prior_attempts_on_question"] == 1
    assert second["user_prior_accuracy"] == pytest.approx(1 / 3)  # (0 + 0.5*2) / (1 + 2)

    row4 = f.loc[4]  # Arrays, medium, Q3, after outcomes 0, 1, 1
    assert row4["user_prior_attempts"] == 3
    assert row4["user_prior_accuracy"] == pytest.approx(0.6)  # (2 + 1) / (3 + 2)
    assert row4["topic_prior_attempts"] == 2
    assert row4["topic_prior_accuracy"] == pytest.approx(0.55)  # (1 + 0.6*2) / (2 + 2)
    assert row4["difficulty_prior_attempts"] == 0
    assert row4["difficulty_prior_accuracy"] == pytest.approx(0.6)  # no medium history yet
    assert row4["recent_accuracy_5"] == pytest.approx(0.64)  # (2 + 0.6*2) / (3 + 2)
    assert row4["topic_prior_solved"] == 1
    assert row4["difficulty_level"] == 2


def test_topic_solved_counts_distinct_questions():
    f = pp.build_features(attempts_frame(HAND_ROWS)).set_index("attempt_id")
    # Q1 was solved twice (rows 2 and 5) but is only ONE distinct solved question.
    assert f.loc[6, "topic_prior_solved"] == 1


# ------------------------------------------------------------------------ leakage
def test_features_have_no_missing_values_and_valid_ranges(features):
    assert not features[pp.FEATURE_COLUMNS].isna().any().any()
    for col in pp.ACCURACY_FEATURES:
        assert features[col].between(0, 1).all()


def test_first_attempt_gets_neutral_defaults(features):
    first = features[features["user_prior_attempts"] == 0]
    assert len(first) == features["user_id"].nunique()
    assert (first["user_prior_accuracy"] == pp.PRIOR_RATE).all()
    assert (first["recent_accuracy_5"] == pp.PRIOR_RATE).all()


def test_an_attempts_own_outcome_never_changes_its_features(synthetic, features):
    """Flip the outcome of each user's LAST attempt. No feature in the table may change,
    because nobody's features depend on a user's final outcome."""
    changed = synthetic.copy()
    last = changed.sort_values(["user_id", "attempted_at", "attempt_id"]).groupby("user_id").tail(1)
    changed.loc[last.index, "is_correct"] = 1 - changed.loc[last.index, "is_correct"]
    after = pp.build_features(changed)
    cols = pp.FEATURE_COLUMNS
    pd.testing.assert_frame_equal(features[cols], after[cols])


def test_future_attempts_never_affect_earlier_rows(synthetic, features):
    """Dropping everything after each user's first 30 attempts must leave those rows' features
    exactly as they were."""
    truncated = synthetic.groupby("user_id").head(30)
    small = pp.build_features(truncated).sort_values("attempt_id").reset_index(drop=True)
    full = features[features["attempt_id"].isin(small["attempt_id"])]
    full = full.sort_values("attempt_id").reset_index(drop=True)
    cols = pp.FEATURE_COLUMNS
    pd.testing.assert_frame_equal(full[cols], small[cols])


# ------------------------------------------------------------ training/serving parity
def test_serving_features_match_training_features(synthetic, features):
    for user_id in (1, 2, 3):
        rows = synthetic[synthetic["user_id"] == user_id].sort_values("attempt_id").reset_index(drop=True)
        history, nxt = rows.iloc[:10], rows.iloc[10]
        served = pp.features_for_next_attempt(
            history, user_id=user_id, question_id=nxt["question_id"],
            topic=nxt["topic"], difficulty=nxt["difficulty"],
        )
        expected = features[features["attempt_id"] == nxt["attempt_id"]][pp.FEATURE_COLUMNS]
        pd.testing.assert_frame_equal(
            served.reset_index(drop=True), expected.reset_index(drop=True)
        )


def test_new_student_with_no_history_gets_defaults():
    empty = pd.DataFrame(columns=pp.REQUIRED_ATTEMPT_COLUMNS)
    row = pp.features_for_next_attempt(
        empty, user_id=5, question_id="ARR-E-001", topic="Arrays", difficulty="easy"
    ).iloc[0]
    assert row["user_prior_attempts"] == 0
    assert row["user_prior_accuracy"] == pytest.approx(0.5)
    assert row["topic_prior_solved"] == 0
    assert row["difficulty_level"] == 1


def test_history_from_another_user_is_rejected():
    history = attempts_frame(HAND_ROWS)  # belongs to user 1
    with pytest.raises(pp.DataValidationError):
        pp.features_for_next_attempt(
            history, user_id=2, question_id="Q9", topic="Arrays", difficulty="easy"
        )


# --------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns=["topic"]),
        lambda d: d.iloc[0:0],
        lambda d: d.assign(topic="Astrology"),
        lambda d: d.assign(difficulty="impossible"),
        lambda d: d.assign(is_correct=2),
        lambda d: d.assign(attempt_id=1),  # duplicates
        lambda d: d.assign(attempted_at=None),
    ],
)
def test_invalid_attempt_tables_rejected(mutate):
    with pytest.raises(pp.DataValidationError):
        pp.validate_attempts(mutate(attempts_frame(HAND_ROWS)))


def test_feature_frame_validation(features):
    ok = features[pp.FEATURE_COLUMNS].head(5)
    assert len(pp.validate_feature_frame(ok)) == 5

    with pytest.raises(pp.DataValidationError):
        pp.validate_feature_frame(ok.drop(columns=["user_prior_accuracy"]))
    with pytest.raises(pp.DataValidationError):
        pp.validate_feature_frame(ok.assign(user_prior_accuracy=1.5))
    with pytest.raises(pp.DataValidationError):
        pp.validate_feature_frame(ok.assign(user_prior_attempts=-1))
    with pytest.raises(pp.DataValidationError):
        pp.validate_feature_frame(ok.assign(topic="Astrology"))
    with pytest.raises(pp.DataValidationError):
        pp.validate_feature_frame(ok.assign(difficulty_level=7))
    with pytest.raises(pp.DataValidationError):
        pp.validate_feature_frame(ok.assign(topic_prior_accuracy="abc"))


# ------------------------------------------------------------------- preprocessor
@pytest.mark.parametrize("scale", [True, False])
def test_preprocessor_output_shape_and_no_nans(features, scale):
    X = features[pp.FEATURE_COLUMNS].head(200).copy()
    X.loc[X.index[0], "recent_accuracy_5"] = np.nan  # missing value must be imputed
    X = pp.validate_feature_frame(X)
    out = pp.make_preprocessor(scale).fit_transform(X)
    assert out.shape == (200, len(pp.NUMERIC_FEATURES) + len(TOPICS))
    assert not np.isnan(out).any()


def test_preprocessor_ignores_topics_it_did_not_see_in_training(features):
    X = pp.validate_feature_frame(features[pp.FEATURE_COLUMNS].head(100))
    only_arrays = X[X["topic"] == "Arrays"]
    if only_arrays.empty:
        pytest.skip("No Arrays rows in this sample")
    pre = pp.make_preprocessor(False).fit(only_arrays)
    assert pre.transform(X).shape[1] == len(pp.NUMERIC_FEATURES) + len(TOPICS)