"""Train, compare and save the success-prediction model.

Run from the project root:
    python -m src.ml.generate_synthetic --students 600 --seed 42     (once)
    python -m src.ml.train
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml.dataset import DatasetError, dataset_fingerprint, load_dataset
from src.ml.evaluate import (
    TEST_FRACTION,
    THRESHOLD,
    VAL_FRACTION,
    classification_metrics,
    metrics_by_history,
    reliability_table,
    split_users,
)
from src.ml.predict import METRICS_FILE, MODEL_FILE
from src.ml.preprocessing import (
    FEATURE_COLUMNS,
    build_features,
    make_preprocessor,
    validate_attempts,
    validate_feature_frame,
)
from src.utils.config import get_settings

ALGORITHM_SHORT = {"logistic_regression": "lr", "random_forest": "rf"}


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def to_jsonable(obj):
    """Round-trip through JSON so numpy types become plain Python values."""
    return json.loads(json.dumps(obj, default=_json_default))


def candidate_pipelines(seed: int, rf_trees: int = 300) -> dict[str, Pipeline]:
    return {
        # lbfgs is deterministic, so no random_state is needed.
        "logistic_regression": Pipeline(
            [("pre", make_preprocessor(scale=True)), ("model", LogisticRegression(max_iter=1000))]
        ),
        # min_samples_leaf keeps leaves large, which avoids overconfident probabilities.
        "random_forest": Pipeline(
            [
                ("pre", make_preprocessor(scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=rf_trees,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def feature_importances(name: str, pipeline: Pipeline) -> dict:
    names = [n.split("__", 1)[1] for n in pipeline.named_steps["pre"].get_feature_names_out()]
    estimator = pipeline.named_steps["model"]
    if name == "logistic_regression":
        values = estimator.coef_[0]
        kind = "standardized coefficient (log-odds change per 1 SD; topic columns are 0/1 indicators)"
    else:
        values = estimator.feature_importances_
        kind = "impurity-based importance (sums to 1; known to favour numeric features)"
    items = sorted(
        ({"feature": n, "value": float(v)} for n, v in zip(names, values)),
        key=lambda d: abs(d["value"]),
        reverse=True,
    )
    return {"kind": kind, "items": items}


def train_and_evaluate(
    attempts: pd.DataFrame,
    *,
    dataset_name: str,
    dataset_type: str,
    seed: int = 42,
    rf_trees: int = 300,
) -> tuple[dict, dict]:
    """Train both candidates, choose by VALIDATION log loss, report TEST metrics.

    Returns (artifact to save with joblib, metrics dict)."""
    if dataset_type not in ("synthetic", "real"):
        raise ValueError("dataset_type must be 'synthetic' or 'real'.")
    attempts = validate_attempts(attempts)
    features = build_features(attempts)
    split = split_users(features["user_id"].unique(), seed)

    X = validate_feature_frame(features[FEATURE_COLUMNS])
    y = features["is_correct"].to_numpy()
    masks = {part: features["user_id"].isin(ids).to_numpy() for part, ids in split.items()}
    y_train, y_val, y_test = y[masks["train"]], y[masks["validation"]], y[masks["test"]]
    if len(np.unique(y_train)) < 2:
        raise ValueError("The training part must contain both correct and incorrect attempts.")

    results = {}
    for name, pipeline in candidate_pipelines(seed, rf_trees).items():
        pipeline.fit(X.loc[masks["train"]], y_train)
        val_p = pipeline.predict_proba(X.loc[masks["validation"]])[:, 1]
        test_p = pipeline.predict_proba(X.loc[masks["test"]])[:, 1]
        results[name] = {
            "pipeline": pipeline,
            "validation": classification_metrics(y_val, val_p),
            "test": classification_metrics(y_test, test_p),
            "test_proba": test_p,
            "importance": feature_importances(name, pipeline),
        }

    selected = min(results, key=lambda n: results[n]["validation"]["log_loss"])
    chosen = results[selected]

    test_rows = features.loc[masks["test"]]
    baseline = classification_metrics(y_test, np.full(len(y_test), float(y_train.mean())))
    oracle = None
    if dataset_type == "synthetic" and "sim_true_prob" in attempts.columns:
        by_id = attempts.set_index("attempt_id")["sim_true_prob"]
        true_p = features["attempt_id"].map(by_id).to_numpy(dtype=float)[masks["test"]]
        if np.isfinite(true_p).all():
            oracle = classification_metrics(y_test, true_p)

    trained_at = datetime.now(timezone.utc)
    version = f"{dataset_name[:30]}-{ALGORITHM_SHORT[selected]}-{trained_at:%Y%m%dT%H%M%S}"
    fingerprint = dataset_fingerprint(attempts)

    notes = [
        "The model was chosen using VALIDATION log loss only; test data was not used to choose.",
        "The split is by student, so no student appears in more than one of train/validation/test.",
        f"Accuracy, precision, recall, F1 and the confusion matrix use a {THRESHOLD} threshold.",
    ]
    if dataset_type == "synthetic":
        notes.append(
            "The dataset is SYNTHETIC (simulated). Scores show how well the model recovers the "
            "simulation's rules, NOT how well it predicts real students."
        )

    artifact = {
        "pipeline": chosen["pipeline"],
        "model_version": version,
        "algorithm": selected,
        "feature_columns": list(FEATURE_COLUMNS),
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "dataset_fingerprint": fingerprint,
        "seed": seed,
        "trained_at_utc": trained_at.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    metrics = {
        "model_version": version,
        "algorithm": selected,
        "selection_rule": "lowest validation log loss",
        "trained_at_utc": trained_at.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "feature_columns": list(FEATURE_COLUMNS),
        "dataset": {
            "name": dataset_name,
            "type": dataset_type,
            "rows": fingerprint["rows"],
            "users": fingerprint["users"],
            "correct_rate": fingerprint["correct"] / fingerprint["rows"],
        },
        "split": {
            "method": "by student (group-aware)",
            "seed": seed,
            "validation_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "users": split,
            "rows": {part: int(mask.sum()) for part, mask in masks.items()},
        },
        "candidates": {
            name: {
                "validation": r["validation"],
                "test": r["test"],
                "feature_importance": r["importance"],
            }
            for name, r in results.items()
        },
        "selected": {
            "test": chosen["test"],
            "calibration": reliability_table(y_test, chosen["test_proba"]),
            "by_history": metrics_by_history(
                test_rows["user_prior_attempts"].to_numpy(), y_test, chosen["test_proba"]
            ),
        },
        "baseline_constant_rate": baseline,
        "oracle_diagnostic": oracle,
        "notes": notes,
    }
    return artifact, metrics


def save_artifacts(artifact: dict, metrics: dict, models_dir: Path) -> tuple[Path, Path]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / MODEL_FILE
    metrics_file = models_dir / METRICS_FILE
    temp = model_file.with_suffix(".tmp")
    joblib.dump(artifact, temp)
    os.replace(temp, model_file)  # never leaves a half-written model behind
    metrics_file.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    return model_file, metrics_file


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(metrics: dict) -> None:
    d, split = metrics["dataset"], metrics["split"]
    users = {k: len(v) for k, v in split["users"].items()}
    print(f"Dataset: {d['name']} ({d['type'].upper()}), {d['rows']} rows, {d['users']} students")
    print(
        f"Split by student: train {users['train']} / validation {users['validation']} / test {users['test']} students"
        f" ({split['rows']['train']} / {split['rows']['validation']} / {split['rows']['test']} rows)"
    )
    print()
    print(f"{'model':<24}{'val_logloss':>12}{'test_auc':>10}{'test_acc':>10}{'test_f1':>9}{'test_brier':>12}")
    for name, c in metrics["candidates"].items():
        t = c["test"]
        print(
            f"{name:<24}{c['validation']['log_loss']:>12.4f}{_fmt(t['roc_auc']):>10}"
            f"{t['accuracy']:>10.3f}{t['f1']:>9.3f}{t['brier']:>12.4f}"
        )
    b = metrics["baseline_constant_rate"]
    print(f"{'constant-rate baseline':<24}{'':>12}{_fmt(b['roc_auc']):>10}{b['accuracy']:>10.3f}{b['f1']:>9.3f}{b['brier']:>12.4f}")
    o = metrics["oracle_diagnostic"]
    if o:
        print(f"{'true-probability ceiling':<24}{'':>12}{_fmt(o['roc_auc']):>10}{o['accuracy']:>10.3f}{o['f1']:>9.3f}{o['brier']:>12.4f}  (synthetic diagnostic)")
    print()
    print(f"Selected: {metrics['algorithm']}  (lowest validation log loss)")
    cm = metrics["selected"]["test"]["confusion_matrix"]
    print(f"Test confusion matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"Calibration error (ECE) on test: {metrics['selected']['calibration']['ece']:.3f}")
    print("Test performance by earlier-history length:")
    for row in metrics["selected"]["by_history"]:
        print(f"  {row['history']:>5} attempts: n={row['n']:<6} AUC={_fmt(row['roc_auc'])}  accuracy={row['accuracy']:.3f}")
    for note in metrics["notes"]:
        print(f"* {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and compare the success-prediction models.")
    parser.add_argument("--data", default=str(get_settings().ml_data_dir / "synthetic_attempts.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-db", action="store_true", help="Do not register the model in the database.")
    args = parser.parse_args()

    try:
        attempts, meta = load_dataset(args.data)
    except DatasetError as exc:
        print(f"Error: {exc}")
        return 1

    print("Training (this can take a minute)...")
    artifact, metrics = train_and_evaluate(
        attempts, dataset_name=meta["dataset_name"], dataset_type=meta["dataset_type"], seed=args.seed
    )
    model_file, metrics_file = save_artifacts(artifact, metrics, get_settings().models_dir)
    print_summary(metrics)
    print(f"\nSaved model:   {model_file}")
    print(f"Saved metrics: {metrics_file}")

    if not args.no_db:
        try:
            from src.database import operations as ops
            from src.database.connection import init_db, session_scope

            init_db()
            with session_scope() as session:
                ops.register_model(
                    session,
                    model_version=metrics["model_version"],
                    algorithm=metrics["algorithm"],
                    trained_at=datetime.fromisoformat(metrics["trained_at_utc"]).replace(tzinfo=None),
                    dataset_name=meta["dataset_name"],
                    dataset_type=meta["dataset_type"],
                    n_train=metrics["split"]["rows"]["train"],
                    n_test=metrics["split"]["rows"]["test"],
                    metrics=to_jsonable(metrics),
                )
            print("Registered in the database (model_metadata).")
        except Exception as exc:
            print(f"Model files were saved, but database registration failed ({type(exc).__name__}).")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())