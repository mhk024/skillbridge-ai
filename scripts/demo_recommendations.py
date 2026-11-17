"""Show how recommendations change with performance, using the real question bank.

    python -m scripts.demo_recommendations
"""

import pandas as pd

from src.assessment.question_service import load_question_bank
from src.ml.predict import ModelUnavailableError, load_model
from src.ml.recommend import make_predictor, recommend
from src.utils.config import get_settings

SCENARIOS = [
    ("New student (no history)", []),
    ("Struggling with Arrays (5 failed attempts)", [("ARR-E-001", 0)] * 5),
    ("Strong in Arrays (3 correct attempts)", [("ARR-E-001", 1)] * 3),
    ("Recovering in Arrays (4 failed, then 4 correct)", [("ARR-E-001", 0)] * 4 + [("ARR-E-001", 1)] * 4),
    ("Struggling in Arrays and has started Sorting", [("ARR-E-001", 0)] * 4 + [("SORT-E-001", 0)]),
]


def build_history(rows, by_id):
    if not rows:
        return None
    start = pd.Timestamp("2026-01-01 09:00:00")
    return pd.DataFrame(
        [
            {
                "attempt_id": i + 1, "user_id": 1, "question_id": qid,
                "topic": by_id[qid]["topic"], "difficulty": by_id[qid]["difficulty"],
                "attempted_at": start + pd.Timedelta(minutes=i), "is_correct": outcome,
            }
            for i, (qid, outcome) in enumerate(rows)
        ]
    )


def main() -> None:
    bank = load_question_bank(get_settings().questions_dir)
    by_id = {q["id"]: q for q in bank}
    try:
        model = load_model()
        print(f"Using model {model.version} (dataset: {model.dataset_type.upper()}).")
    except ModelUnavailableError as exc:
        model = None
        print(f"No model estimates ({exc})")

    for title, rows in SCENARIOS:
        hist = build_history(rows, by_id)
        predict = make_predictor(model, hist, 1) if model else None
        recs = recommend(hist, bank, predict=predict, n=5)
        print(f"\n=== {title} ===")
        for i, r in enumerate(recs, start=1):
            p = "n/a" if r.predicted_prob is None else f"{r.predicted_prob:.0%}"
            print(f"{i}. {r.question_id:<11} {r.topic:<20} {r.difficulty:<7} {r.strategy:<12} est={p}")
            print(f"     {r.reason}")


if __name__ == "__main__":
    main()