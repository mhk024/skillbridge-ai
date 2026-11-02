"""Data validation and leakage-safe feature engineering.

Raw table: ONE ROW PER ATTEMPT with the columns in REQUIRED_ATTEMPT_COLUMNS.
Target   : is_correct (1 = the attempt passed all tests, 0 = it did not).

LEAKAGE RULE: every feature for an attempt is computed only from that same student's
STRICTLY EARLIER attempts. An attempt's own outcome is never used for its own features,
and later attempts never influence earlier rows.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.assessment.question_service import TOPICS
from src.database.models import DIFFICULTIES

REQUIRED_ATTEMPT_COLUMNS = [
    "attempt_id", "user_id", "question_id", "topic", "difficulty", "attempted_at", "is_correct",
]

# Smoothing: with no history an accuracy defaults to PRIOR_RATE; PRIOR_STRENGTH acts like
# that many "imaginary" attempts. Fixed constants, so nothing is learned from other users.
PRIOR_RATE = 0.5
PRIOR_STRENGTH = 2.0
RECENT_WINDOW = 5
DIFFICULTY_LEVEL = {"easy": 1, "medium": 2, "hard": 3}

ACCURACY_FEATURES = [
    "user_prior_accuracy", "recent_accuracy_5", "topic_prior_accuracy", "difficulty_prior_accuracy",
]
COUNT_FEATURES = [
    "user_prior_attempts", "topic_prior_attempts", "topic_prior_solved",
    "difficulty_prior_attempts", "prior_attempts_on_question",
]
NUMERIC_FEATURES = [
    "user_prior_attempts", "user_prior_accuracy", "recent_accuracy_5",
    "topic_prior_attempts", "topic_prior_accuracy", "topic_prior_solved",
    "difficulty_prior_attempts", "difficulty_prior_accuracy",
    "difficulty_level", "prior_attempts_on_question",
]
CATEGORICAL_FEATURES = ["topic"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class DataValidationError(ValueError):
    """The attempts table or a feature frame is not in the expected shape."""


# --------------------------------------------------------------------- validation
def validate_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    """Check the raw attempts table and return a cleaned copy."""
    if not isinstance(attempts, pd.DataFrame):
        raise DataValidationError("Attempts must be a pandas DataFrame.")
    missing = [c for c in REQUIRED_ATTEMPT_COLUMNS if c not in attempts.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}.")
    if attempts.empty:
        raise DataValidationError("The attempts table has no rows.")

    df = attempts.copy()
    null_cols = [c for c in REQUIRED_ATTEMPT_COLUMNS if df[c].isna().any()]
    if null_cols:
        raise DataValidationError(f"Missing values in column(s): {', '.join(null_cols)}.")

    try:
        df["attempt_id"] = pd.to_numeric(df["attempt_id"], errors="raise").astype("int64")
        df["user_id"] = pd.to_numeric(df["user_id"], errors="raise").astype("int64")
        df["attempted_at"] = pd.to_datetime(df["attempted_at"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Could not read ids or timestamps: {exc}") from exc
    if df["attempted_at"].dt.tz is not None:
        df["attempted_at"] = df["attempted_at"].dt.tz_convert("UTC").dt.tz_localize(None)

    if not df["is_correct"].isin([0, 1, True, False]).all():
        raise DataValidationError("is_correct must contain only 0/1 (or True/False).")
    df["is_correct"] = df["is_correct"].astype("int64")

    bad_topics = sorted(set(df.loc[~df["topic"].isin(TOPICS), "topic"].astype(str)))
    if bad_topics:
        raise DataValidationError(f"Unknown topic(s): {bad_topics[:3]}")
    bad_diff = sorted(set(df.loc[~df["difficulty"].isin(DIFFICULTIES), "difficulty"].astype(str)))
    if bad_diff:
        raise DataValidationError(f"Unknown difficulty value(s): {bad_diff[:3]}")
    if df["attempt_id"].duplicated().any():
        raise DataValidationError("attempt_id values must be unique.")
    return df


def validate_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Check a feature table before prediction. Missing numeric values are allowed
    (the preprocessor imputes them); wrong columns, bad ranges and unknown topics are not."""
    if not isinstance(features, pd.DataFrame):
        raise DataValidationError("Features must be a pandas DataFrame.")
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise DataValidationError(f"Missing feature column(s): {', '.join(missing)}.")
    if features.empty:
        raise DataValidationError("The feature table has no rows.")

    out = features[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(out[col], errors="coerce")
        if (out[col].notna() & values.isna()).any():
            raise DataValidationError(f"Column '{col}' must be numeric.")
        if np.isinf(values).any():
            raise DataValidationError(f"Column '{col}' contains infinite values.")
        if col in ACCURACY_FEATURES and ((values < 0) | (values > 1)).any():
            raise DataValidationError(f"Column '{col}' must be between 0 and 1.")
        if col in COUNT_FEATURES and (values < 0).any():
            raise DataValidationError(f"Column '{col}' cannot be negative.")
        out[col] = values.astype("float64")
    if not out["difficulty_level"].dropna().isin([1, 2, 3]).all():
        raise DataValidationError("difficulty_level must be 1, 2 or 3.")
    if not out["topic"].isin(TOPICS).all():
        raise DataValidationError("Unknown or missing topic in the feature table.")
    return out


# ------------------------------------------------------------------ feature engineering
def _smoothed(correct, attempts, prior):
    """Accuracy shrunk toward `prior` (a number or a Series) so tiny samples are not extreme."""
    return (correct + prior * PRIOR_STRENGTH) / (attempts + PRIOR_STRENGTH)


def build_features(attempts: pd.DataFrame) -> pd.DataFrame:
    """One feature row per attempt, using only that student's earlier attempts.

    Rows are ordered by (user_id, attempted_at, attempt_id). The returned frame has
    attempt_id, user_id, attempted_at, is_correct (the target) and FEATURE_COLUMNS.
    """
    df = validate_attempts(attempts)
    df = df.sort_values(["user_id", "attempted_at", "attempt_id"], kind="mergesort")
    df = df.reset_index(drop=True)
    y = df["is_correct"]

    # --- whole-history features
    by_user = df.groupby("user_id", sort=False)
    user_attempts = by_user.cumcount()  # attempts BEFORE this one
    user_correct = by_user["is_correct"].cumsum() - y  # correct answers BEFORE this one
    user_acc = _smoothed(user_correct, user_attempts, PRIOR_RATE)

    # --- last RECENT_WINDOW attempts (difference of cumulative counts)
    window_start = user_correct.groupby(df["user_id"], sort=False).shift(RECENT_WINDOW).fillna(0)
    recent_correct = user_correct - window_start
    recent_n = user_attempts.clip(upper=RECENT_WINDOW)
    recent_acc = _smoothed(recent_correct, recent_n, user_acc)

    # --- same topic (shrunk toward the student's own overall accuracy)
    by_topic = df.groupby(["user_id", "topic"], sort=False)
    topic_attempts = by_topic.cumcount()
    topic_correct = by_topic["is_correct"].cumsum() - y
    topic_acc = _smoothed(topic_correct, topic_attempts, user_acc)

    # --- same difficulty
    by_diff = df.groupby(["user_id", "difficulty"], sort=False)
    diff_attempts = by_diff.cumcount()
    diff_correct = by_diff["is_correct"].cumsum() - y
    diff_acc = _smoothed(diff_correct, diff_attempts, user_acc)

    # --- same question, and distinct questions already solved in this topic
    by_question = df.groupby(["user_id", "question_id"], sort=False)
    question_attempts = by_question.cumcount()
    question_correct = by_question["is_correct"].cumsum() - y
    first_solve = ((y == 1) & (question_correct == 0)).astype("int64")
    topic_solved = (
        first_solve.groupby([df["user_id"], df["topic"]], sort=False).cumsum() - first_solve
    )

    out = pd.DataFrame(
        {
            "attempt_id": df["attempt_id"],
            "user_id": df["user_id"],
            "attempted_at": df["attempted_at"],
            "is_correct": y,
            "user_prior_attempts": user_attempts,
            "user_prior_accuracy": user_acc,
            "recent_accuracy_5": recent_acc,
            "topic_prior_attempts": topic_attempts,
            "topic_prior_accuracy": topic_acc,
            "topic_prior_solved": topic_solved,
            "difficulty_prior_attempts": diff_attempts,
            "difficulty_prior_accuracy": diff_acc,
            "difficulty_level": df["difficulty"].map(DIFFICULTY_LEVEL),
            "prior_attempts_on_question": question_attempts,
            "topic": df["topic"],
        }
    )
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype("float64")
    return out


def features_for_next_attempt(
    history: pd.DataFrame | None,
    *,
    user_id: int,
    question_id: str,
    topic: str,
    difficulty: str,
) -> pd.DataFrame:
    """Features for a student's NEXT attempt, using the SAME code as training.

    A placeholder attempt is appended after the history. Its outcome (0) cannot influence
    its own features, so this is identical to how the training rows were built.
    """
    if history is None or len(history) == 0:
        base = None
        next_id = 1
        stamp = pd.Timestamp.now()
    else:
        base = validate_attempts(history)
        if (base["user_id"] != user_id).any():
            raise DataValidationError("History contains attempts from a different user.")
        next_id = int(base["attempt_id"].max()) + 1
        stamp = base["attempted_at"].max() + pd.Timedelta(seconds=1)

    placeholder = pd.DataFrame(
        [
            {
                "attempt_id": next_id,
                "user_id": user_id,
                "question_id": question_id,
                "topic": topic,
                "difficulty": difficulty,
                "attempted_at": stamp,
                "is_correct": 0,
            }
        ]
    )
    attempts = placeholder if base is None else pd.concat(
        [base[REQUIRED_ATTEMPT_COLUMNS], placeholder], ignore_index=True
    )
    features = build_features(attempts)
    row = features[features["attempt_id"] == next_id]
    return row[FEATURE_COLUMNS].reset_index(drop=True)


# ----------------------------------------------------------------------- preprocessor
def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Imputation (+ optional scaling) for numbers, fixed-category one-hot for topic.

    scale=True for Logistic Regression; scale=False for tree models (no need)."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[list(TOPICS)], handle_unknown="ignore", sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )