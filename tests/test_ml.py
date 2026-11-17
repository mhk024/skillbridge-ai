import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.database import operations as ops
from src.database.connection import session_scope
from src.database.models import ModelMetadata
from src.ml import evaluate as ev
from src.ml import predict as pr
from src.ml import preprocessing as pp
from src.ml import train as tr
from src.ml.dataset import DatasetError, dataset_fingerprint, load_dataset
from src.ml.generate_synthetic import generate_attempts, save_dataset


@pytest.fixture(scope="module")
def attempts():
    return generate_attempts(n_students=120, seed=3)


@pytest.fixture(scope="module")
def trained(attempts):
    return tr.train_and_evaluate(
        attempts, dataset_name="test_synthetic", dataset_type="synthetic", seed=11, rf_trees=40
    )


@pytest.fixture(scope="module")
def model(trained):
    return pr.model_from_artifact(trained[0])


def make_history(outcome: int, n: int = 20) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01 09:00:00")
    return pd.DataFrame(
        [
            {
                "attempt_id": i + 1, "user_id": 1, "question_id": f"H{i}", "topic": "Arrays",
                "difficulty": "easy", "attempted_at": base + pd.Timedelta(minutes=i),
                "is_correct": outcome,
            }
            for i in range(n)
        ]
    )


# ----------------------------------------------------------------------- the split
def test_split_is_disjoint_complete_and_deterministic():
    ids = list(range(1, 201))
    a, b = ev.split_users(ids, seed=5), ev.split_users(ids, seed=5)
    assert a == b
    parts = [set(a[k]) for k in ("train", "validation", "test")]
    assert set().union(*parts) == set(ids)
    assert sum(len(p) for p in parts) == 200
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert len(a["test"]) == 30 and len(a["validation"]) == 30


def test_different_seed_gives_different_split():
    ids = list(range(1, 101))
    assert ev.split_users(ids, 1)["test"] != ev.split_users(ids, 2)["test"]


def test_too_few_students_rejected():
    with pytest.raises(ValueError):
        ev.split_users(range(5), seed=1)


# ----------------------------------------------------------------------- the metrics
def test_classification_metrics_match_hand_calculation():
    m = ev.classification_metrics([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6])
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(0.1925)
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.4, 0.4])))


def test_single_class_gives_no_auc_but_other_metrics():
    m = ev.classification_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert m["roc_auc"] is None and m["accuracy"] == 1.0


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        ev.classification_metrics([], [])
    with pytest.raises(ValueError):
        ev.classification_metrics([1, 0], [0.5])


def test_reliability_table_perfect_and_bad_calibration():
    perfect = ev.reliability_table([1] + [0] * 9, [0.1] * 10)
    assert perfect["ece"] == pytest.approx(0.0, abs=1e-9)
    bad = ev.reliability_table([0] * 10, [0.95] * 10)
    assert bad["ece"] == pytest.approx(0.95)
    assert sum(b["count"] for b in bad["bins"]) == 10


def test_metrics_by_history_buckets():
    counts = np.array([0, 3, 5, 19, 20, 50])
    rows = ev.metrics_by_history(counts, [1, 0, 1, 0, 1, 0], [0.8, 0.3, 0.7, 0.4, 0.6, 0.2])
    assert [r["history"] for r in rows] == ["0-4", "5-19", "20+"]
    assert [r["n"] for r in rows] == [2, 2, 2]


# ----------------------------------------------------------------- leakage / schema
def test_label_and_diagnostic_columns_are_not_features():
    assert "is_correct" not in pp.FEATURE_COLUMNS
    assert "sim_true_prob" not in pp.FEATURE_COLUMNS


# -------------------------------------------------------------------------- training
def test_training_produces_consistent_artifact_and_metrics(trained):
    artifact, m = trained
    assert artifact["feature_columns"] == pp.FEATURE_COLUMNS
    assert m["algorithm"] in m["candidates"] and set(m["candidates"]) == {"logistic_regression", "random_forest"}

    users = {k: set(v) for k, v in m["split"]["users"].items()}
    assert users["train"].isdisjoint(users["validation"])
    assert users["train"].isdisjoint(users["test"])
    assert users["validation"].isdisjoint(users["test"])
    assert m["split"]["rows"]["test"] == m["selected"]["test"]["n"]

    best = min(m["candidates"], key=lambda n: m["candidates"][n]["validation"]["log_loss"])
    assert m["algorithm"] == best
    assert m["selected"]["test"] == m["candidates"][best]["test"]
    assert m["dataset"]["type"] == "synthetic"
    assert m["oracle_diagnostic"] is not None
    assert any("SYNTHETIC" in note for note in m["notes"])
    # Sanity check only: better than chance on simulated data. NOT evidence of real-world accuracy.
    assert m["selected"]["test"]["roc_auc"] > 0.6


def test_metrics_are_json_serializable(trained):
    json.dumps(tr.to_jsonable(trained[1]))


def test_training_rejects_bad_inputs(attempts):
    with pytest.raises(ValueError):
        tr.train_and_evaluate(attempts, dataset_name="x", dataset_type="invented")
    with pytest.raises(ValueError):
        tr.train_and_evaluate(generate_attempts(5, 1), dataset_name="x", dataset_type="synthetic")


def test_saved_model_evaluation_reproduces_test_metrics(trained, model, attempts):
    artifact, metrics = trained
    again = ev.evaluate_saved_model(model, attempts, seed=artifact["seed"])
    saved = metrics["selected"]["test"]
    for key in ("accuracy", "roc_auc", "log_loss", "brier"):
        assert again[key] == pytest.approx(saved[key], abs=1e-9)


# --------------------------------------------------------------- saving and loading
def test_save_and_load_roundtrip(trained, model, attempts, tmp_path):
    artifact, metrics = trained
    model_file, metrics_file = tr.save_artifacts(artifact, metrics, tmp_path)
    loaded = pr.load_model(model_file)
    assert loaded.version == artifact["model_version"]
    X = pp.build_features(attempts)[pp.FEATURE_COLUMNS].head(50)
    np.testing.assert_allclose(pr.predict_proba(loaded, X), pr.predict_proba(model, X))
    assert pr.load_metrics(metrics_file)["model_version"] == artifact["model_version"]
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(pr.ModelUnavailableError) as exc:
        pr.load_model(tmp_path / "nope.joblib")
    assert "python -m src.ml.train" in str(exc.value)


def test_corrupt_model_file_gives_clean_error(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a model")
    with pytest.raises(pr.ModelUnavailableError):
        pr.load_model(bad)


def test_incomplete_and_wrong_schema_artifacts_rejected(trained, tmp_path):
    artifact, _ = trained
    incomplete = tmp_path / "incomplete.joblib"
    joblib.dump({"pipeline": artifact["pipeline"]}, incomplete)
    with pytest.raises(pr.ModelUnavailableError):
        pr.load_model(incomplete)

    wrong = tmp_path / "wrong.joblib"
    joblib.dump(dict(artifact, feature_columns=["a", "b"]), wrong)
    with pytest.raises(pr.ModelUnavailableError) as exc:
        pr.load_model(wrong)
    assert "feature" in str(exc.value).lower()


def test_sklearn_version_mismatch_gives_warning_not_failure(trained):
    loaded = pr.model_from_artifact(dict(trained[0], sklearn_version="0.0.1"))
    assert loaded.version_warning is not None


def test_load_metrics_missing_file_returns_none(tmp_path):
    assert pr.load_metrics(tmp_path / "missing.json") is None


# --------------------------------------------------------------------- predictions
def test_predictions_are_probabilities_and_missing_values_handled(model, attempts):
    X = pp.build_features(attempts)[pp.FEATURE_COLUMNS].head(30).copy()
    X.loc[X.index[0], "recent_accuracy_5"] = np.nan
    p = pr.predict_proba(model, X)
    assert p.shape == (30,)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_predict_rejects_invalid_feature_frames(model, attempts):
    X = pp.build_features(attempts)[pp.FEATURE_COLUMNS].head(5)
    with pytest.raises(pp.DataValidationError):
        pr.predict_proba(model, X.drop(columns=["topic"]))
    with pytest.raises(pp.DataValidationError):
        pr.predict_proba(model, X.assign(user_prior_accuracy=2.0))


def test_stronger_history_gives_higher_estimate(model):
    def estimate(outcome):
        f = pp.features_for_next_attempt(
            make_history(outcome), user_id=1, question_id="NEW", topic="Arrays", difficulty="medium"
        )
        return pr.predict_proba(model, f)[0]

    assert estimate(1) > estimate(0)


# ----------------------------------------------------------- database integration
def _make_user(engine, name):
    with session_scope(engine) as s:
        return ops.create_user(s, username=name, email=f"{name}@example.com", password_hash="x" * 60).id


def _submit(engine, uid, aid, qid, passed, total=5):
    with session_scope(engine) as s:
        ops.record_submission(
            s, user_id=uid, assessment_id=aid, question_id=qid,
            code="x", passed_tests=passed, total_tests=total,
        )


def test_history_from_db_matches_expected_schema_and_is_isolated(seeded_engine):
    alice, bob = _make_user(seeded_engine, "alice"), _make_user(seeded_engine, "bob")
    with session_scope(seeded_engine) as s:
        a_id = ops.create_assessment(s, user_id=alice, mode="judge").id
        b_id = ops.create_assessment(s, user_id=bob, mode="judge").id
    _submit(seeded_engine, alice, a_id, "ARR-E-001", 5)
    _submit(seeded_engine, alice, a_id, "STR-E-001", 0)
    _submit(seeded_engine, bob, b_id, "ARR-E-001", 5)

    with session_scope(seeded_engine) as s:
        history = pr.history_from_db(s, alice)
        empty = pr.history_from_db(s, 99999)
    assert list(history.columns) == pp.REQUIRED_ATTEMPT_COLUMNS
    assert len(history) == 2 and set(history["user_id"]) == {alice}
    assert set(history["topic"]) == {"Arrays", "Strings"}
    pp.validate_attempts(history)  # raises if the schema were wrong
    assert len(empty) == 0


def test_prediction_from_database_history_matches_direct_computation(seeded_engine, model):
    alice, carol = _make_user(seeded_engine, "alice"), _make_user(seeded_engine, "carol")
    with session_scope(seeded_engine) as s:
        a_id = ops.create_assessment(s, user_id=alice, mode="judge").id
    _submit(seeded_engine, alice, a_id, "ARR-E-001", 5)
    _submit(seeded_engine, alice, a_id, "STR-E-001", 0)

    kwargs = dict(question_id="DP-E-001", topic="Dynamic Programming", difficulty="easy")
    with session_scope(seeded_engine) as s:
        p_db = pr.predict_for_next_attempt(s, model, user_id=alice, **kwargs)
        history = pr.history_from_db(s, alice)
        p_new_user = pr.predict_for_next_attempt(s, model, user_id=carol, **kwargs)
    direct = pr.predict_proba(model, pp.features_for_next_attempt(history, user_id=alice, **kwargs))[0]
    assert p_db == pytest.approx(direct)
    assert 0.0 <= p_new_user <= 1.0  # a student with no history still gets an estimate


def test_register_model_activates_latest_and_rejects_duplicates(engine, trained):
    metrics = tr.to_jsonable(trained[1])

    def register(version, dataset_type="synthetic"):
        with session_scope(engine) as s:
            ops.register_model(
                s, model_version=version, algorithm="random_forest", trained_at=datetime(2026, 1, 1),
                dataset_name="d", dataset_type=dataset_type, n_train=10, n_test=5, metrics=metrics,
            )

    register("v1")
    register("v2")
    with session_scope(engine) as s:
        assert ops.get_active_model_metadata(s).model_version == "v2"
        rows = list(s.scalars(select(ModelMetadata)))
        assert len(rows) == 2 and sum(r.is_active for r in rows) == 1
        assert ops.get_active_model_metadata(s).metrics_json["model_version"] == trained[1]["model_version"]
    with pytest.raises(ValueError):
        register("v2")
    with pytest.raises(ValueError):
        register("v3", dataset_type="magic")


# ------------------------------------------------------------------------ datasets
def test_load_dataset_requires_valid_metadata(attempts, tmp_path):
    path = save_dataset(attempts, tmp_path, seed=3)
    df, meta = load_dataset(path)
    assert meta["dataset_type"] == "synthetic" and len(df) == len(attempts)
    assert dataset_fingerprint(df) == dataset_fingerprint(attempts)

    meta_file = tmp_path / "synthetic_attempts.meta.json"
    meta_file.write_text(json.dumps({"dataset_name": "x", "dataset_type": "magic"}), encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(path)
    meta_file.unlink()
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_missing_dataset_file_gives_clear_error(tmp_path):
    with pytest.raises(DatasetError) as exc:
        load_dataset(tmp_path / "nope.csv")
    assert "generate_synthetic" in str(exc.value)