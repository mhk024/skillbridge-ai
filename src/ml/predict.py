"""Load the trained model and make predictions.

SECURITY NOTE: joblib files are Python pickles. Only load model files produced by this
project's own training script; never load a file from an untrusted source.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Question, Submission
from src.ml.preprocessing import (
    FEATURE_COLUMNS,
    features_for_next_attempt,
    validate_feature_frame,
)
from src.utils.config import get_settings

logger = logging.getLogger("skillbridge")

MODEL_FILE = "skillbridge_model.joblib"
METRICS_FILE = "model_metrics.json"
TRAIN_COMMAND = "python -m src.ml.train"
_REQUIRED_KEYS = (
    "pipeline", "model_version", "algorithm", "feature_columns", "dataset_type", "dataset_name",
)


class ModelUnavailableError(Exception):
    """No usable model. Messages are safe to show to users."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: object
    version: str
    algorithm: str
    dataset_type: str
    dataset_name: str
    metadata: dict
    version_warning: str | None = None


def model_path(models_dir: Path | None = None) -> Path:
    return Path(models_dir or get_settings().models_dir) / MODEL_FILE


def metrics_path(models_dir: Path | None = None) -> Path:
    return Path(models_dir or get_settings().models_dir) / METRICS_FILE


def model_file_mtime() -> float | None:
    """Modification time of the model file (used as a cache key), or None if missing."""
    path = model_path()
    return path.stat().st_mtime if path.exists() else None


def model_from_artifact(artifact) -> LoadedModel:
    """Validate a loaded artifact and wrap it. Raises ModelUnavailableError if unusable."""
    if not isinstance(artifact, dict) or any(k not in artifact for k in _REQUIRED_KEYS):
        raise ModelUnavailableError(
            f"The model file is incomplete or from an older version. Retrain with: {TRAIN_COMMAND}"
        )
    if list(artifact["feature_columns"]) != FEATURE_COLUMNS:
        raise ModelUnavailableError(
            f"The model was trained with a different feature set than this app uses. "
            f"Retrain with: {TRAIN_COMMAND}"
        )
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise ModelUnavailableError(f"The model file does not contain a usable model. Retrain with: {TRAIN_COMMAND}")

    warning = None
    trained_with = artifact.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        warning = (
            f"Model was trained with scikit-learn {trained_with} but this system runs "
            f"{sklearn.__version__}. Retrain if results look wrong."
        )
    return LoadedModel(
        pipeline=artifact["pipeline"],
        version=str(artifact["model_version"]),
        algorithm=str(artifact["algorithm"]),
        dataset_type=str(artifact["dataset_type"]),
        dataset_name=str(artifact["dataset_name"]),
        metadata={k: v for k, v in artifact.items() if k != "pipeline"},
        version_warning=warning,
    )


def load_model(path: Path | None = None) -> LoadedModel:
    path = Path(path) if path else model_path()
    if not path.exists():
        raise ModelUnavailableError(
            "No trained model found. Create the data and train one with: "
            f"python -m src.ml.generate_synthetic  then  {TRAIN_COMMAND}"
        )
    try:
        artifact = joblib.load(path)
    except Exception as exc:  # corrupt or incompatible file; details go to the log only
        logger.exception("Could not read model file %s", path)
        raise ModelUnavailableError(
            f"The model file could not be read. Retrain with: {TRAIN_COMMAND}"
        ) from exc
    return model_from_artifact(artifact)


def load_metrics(path: Path | None = None) -> dict | None:
    """Metrics saved at training time, or None if unavailable."""
    path = Path(path) if path else metrics_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def predict_proba(model: LoadedModel, features: pd.DataFrame) -> np.ndarray:
    """Probability that each row's attempt will be correct."""
    X = validate_feature_frame(features)  # raises DataValidationError on bad input
    try:
        proba = model.pipeline.predict_proba(X)[:, 1]
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise ModelUnavailableError("The model could not make a prediction right now.") from exc
    return np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)


def history_from_db(session: Session, user_id: int) -> pd.DataFrame:
    """This user's saved attempts in the exact table shape the feature code expects."""
    rows = session.execute(
        select(
            Submission.id,
            Submission.user_id,
            Submission.question_id,
            Question.topic,
            Question.difficulty,
            Submission.submitted_at,
            Submission.is_correct,
        )
        .join(Question, Question.id == Submission.question_id)
        .where(Submission.user_id == user_id)
        .order_by(Submission.submitted_at, Submission.id)
    ).all()
    df = pd.DataFrame(
        rows,
        columns=["attempt_id", "user_id", "question_id", "topic", "difficulty",
                 "attempted_at", "is_correct"],
    )
    df["is_correct"] = df["is_correct"].astype("int64")
    return df


def predict_for_next_attempt(
    session: Session,
    model: LoadedModel,
    *,
    user_id: int,
    question_id: str,
    topic: str,
    difficulty: str,
) -> float:
    """Estimated probability that this user's NEXT attempt on the question is correct."""
    history = history_from_db(session, user_id)
    features = features_for_next_attempt(
        history, user_id=user_id, question_id=question_id, topic=topic, difficulty=difficulty
    )
    return float(predict_proba(model, features)[0])