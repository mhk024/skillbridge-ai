"""Evaluation helpers (student-grouped split, metrics, calibration) and a re-evaluation command.

    python -m src.ml.evaluate
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.ml.dataset import DatasetError, dataset_fingerprint, load_dataset
from src.ml.predict import (
    LoadedModel,
    ModelUnavailableError,
    load_metrics,
    load_model,
    predict_proba,
)
from src.ml.preprocessing import FEATURE_COLUMNS, build_features, validate_feature_frame
from src.utils.config import get_settings

THRESHOLD = 0.5
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
PROBA_CLIP = 1e-6
HISTORY_BUCKETS = (("0-4", 0, 4), ("5-19", 5, 19), ("20+", 20, None))


def split_users(
    user_ids, seed: int, val_fraction: float = VAL_FRACTION, test_fraction: float = TEST_FRACTION
) -> dict[str, list[int]]:
    """Divide STUDENTS (not rows) into train / validation / test, reproducibly."""
    users = np.array(sorted({int(u) for u in user_ids}))
    if len(users) < 10:
        raise ValueError("At least 10 students are needed to split by student.")
    shuffled = np.random.default_rng(seed).permutation(users)
    n_test = max(1, int(round(len(users) * test_fraction)))
    n_val = max(1, int(round(len(users) * val_fraction)))
    return {
        "train": sorted(int(u) for u in shuffled[n_test + n_val:]),
        "validation": sorted(int(u) for u in shuffled[n_test:n_test + n_val]),
        "test": sorted(int(u) for u in shuffled[:n_test]),
    }


def classification_metrics(y_true, proba, threshold: float = THRESHOLD) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), PROBA_CLIP, 1 - PROBA_CLIP)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and proba must be non-empty and the same length.")
    pred = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, p)) if has_both_classes else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def reliability_table(y_true, proba, n_bins: int = 10) -> dict:
    """Do predicted probabilities match observed frequencies? Also the expected calibration
    error (ECE): the average gap between predicted and observed, weighted by bin size."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.digitize(p, edges[1:-1])
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred, observed = float(p[mask].mean()), float(y[mask].mean())
        ece += count / len(y) * abs(observed - mean_pred)
        bins.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "count": count,
                "mean_predicted": mean_pred,
                "observed_rate": observed,
            }
        )
    return {"bins": bins, "ece": float(ece)}


def metrics_by_history(history_counts, y_true, proba) -> list[dict]:
    """Performance split by how many earlier attempts the student had (cold start vs warm)."""
    counts = np.asarray(history_counts)
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    rows = []
    for label, low, high in HISTORY_BUCKETS:
        mask = counts >= low
        if high is not None:
            mask &= counts <= high
        if not mask.any():
            continue
        m = classification_metrics(y[mask], p[mask])
        rows.append(
            {
                "history": label,
                "n": m["n"],
                "positive_rate": m["positive_rate"],
                "roc_auc": m["roc_auc"],
                "accuracy": m["accuracy"],
                "brier": m["brier"],
            }
        )
    return rows


def evaluate_saved_model(model: LoadedModel, attempts: pd.DataFrame, seed: int) -> dict:
    """Score a saved model on the held-out test students recovered from (data, seed)."""
    features = build_features(attempts)
    split = split_users(features["user_id"].unique(), seed)
    mask = features["user_id"].isin(split["test"]).to_numpy()
    X = validate_feature_frame(features[FEATURE_COLUMNS]).loc[mask]
    y = features["is_correct"].to_numpy()[mask]
    return classification_metrics(y, predict_proba(model, X))


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) < 1e-9


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate the saved model on its held-out test students.")
    parser.add_argument("--data", default=str(get_settings().ml_data_dir / "synthetic_attempts.csv"))
    args = parser.parse_args()

    try:
        model = load_model()
        attempts, meta = load_dataset(args.data)
    except (ModelUnavailableError, DatasetError) as exc:
        print(f"Error: {exc}")
        return 1

    if dataset_fingerprint(attempts) != model.metadata.get("dataset_fingerprint"):
        print(
            "Error: this dataset is not the one the model was trained on, so the held-out test "
            "students cannot be recovered. Retrain on it with: python -m src.ml.train"
        )
        return 1

    result = evaluate_saved_model(model, attempts, int(model.metadata["seed"]))
    print(f"Model: {model.version} ({model.algorithm})")
    print(f"Dataset: {meta['dataset_name']} ({meta['dataset_type'].upper()}); held-out test rows: {result['n']}")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier"):
        value = result[key]
        print(f"  {key:<10} {'n/a' if value is None else f'{value:.4f}'}")
    cm = result["confusion_matrix"]
    print(f"  confusion matrix (threshold {result['threshold']}): TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")

    saved = ((load_metrics() or {}).get("selected") or {}).get("test")
    if saved is None:
        print("No saved metrics file to compare with.")
        return 0
    same = all(_same(result[k], saved[k]) for k in ("accuracy", "roc_auc", "log_loss", "brier"))
    print("Reproduces the metrics saved at training time: " + ("YES" if same else "NO (investigate)"))
    if meta["dataset_type"] == "synthetic":
        print("Reminder: the data is SYNTHETIC, so these scores are not real-world performance.")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())