"""Load an attempts dataset together with its metadata file."""

import json
from pathlib import Path

import pandas as pd

from src.ml.preprocessing import DataValidationError, validate_attempts

DATASET_TYPES = ("synthetic", "real")


class DatasetError(Exception):
    """The dataset or its metadata could not be loaded. Messages are safe to show."""


def meta_path_for(csv_path: Path) -> Path:
    return Path(csv_path).with_suffix(".meta.json")


def load_dataset(csv_path) -> tuple[pd.DataFrame, dict]:
    """Return (attempts table, metadata). The metadata file is required so a dataset can
    never be silently mislabelled as real or synthetic."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise DatasetError(
            f"Dataset file not found: {csv_path}. "
            "Generate one with: python -m src.ml.generate_synthetic"
        )
    meta_file = meta_path_for(csv_path)
    if not meta_file.exists():
        raise DatasetError(
            f"Metadata file not found: {meta_file}. It must state dataset_name and "
            "dataset_type ('synthetic' or 'real'); the type is never guessed."
        )
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Metadata file could not be read: {meta_file}.") from exc
    if not isinstance(meta, dict) or not meta.get("dataset_name") or meta.get("dataset_type") not in DATASET_TYPES:
        raise DatasetError("Metadata needs a dataset_name and a dataset_type of 'synthetic' or 'real'.")

    try:
        attempts = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError(f"Dataset file could not be read: {csv_path}.") from exc
    try:
        validate_attempts(attempts)
    except DataValidationError as exc:
        raise DatasetError(f"Dataset is invalid: {exc}") from exc
    return attempts, meta


def dataset_fingerprint(attempts: pd.DataFrame) -> dict:
    """Cheap identity check: detects when a different dataset is used for evaluation."""
    return {
        "rows": int(len(attempts)),
        "users": int(attempts["user_id"].nunique()),
        "correct": int(attempts["is_correct"].astype(int).sum()),
    }