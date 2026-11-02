"""Generate a SYNTHETIC coding-attempt dataset for development.

THIS IS NOT REAL STUDENT DATA. It comes from a simulation with hand-chosen assumptions
(see ASSUMPTIONS below). Model scores on it show how well a model recovers those
assumptions, NOT how well it predicts real students.

Run from the project root:
    python -m src.ml.generate_synthetic --students 600 --seed 42
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.assessment.question_service import TOPICS
from src.ml.preprocessing import validate_attempts
from src.utils.config import get_settings

DATASET_NAME = "skillbridge_synthetic_v1"
DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_HARDNESS = {"easy": -1.2, "medium": 0.2, "hard": 1.6}
QUESTIONS_PER_CELL = 6  # synthetic questions per (topic, difficulty)
SLOPE = 1.4  # how sharply skill-minus-hardness turns into success probability
START_DATE = pd.Timestamp("2025-01-01")

ASSUMPTIONS = [
    "Each student has an overall ability (normal) plus per-topic variation.",
    "Skill in a topic grows slightly with practice there (capped at 15 prior attempts).",
    "Topics have different intrinsic hardness; difficulty levels add fixed hardness.",
    "Success probability = logistic(1.4 * (skill - hardness)); each question has its own small offset.",
    "Students choose easier or harder questions depending on their recent success.",
    "With 20% probability a student retries a question they failed and have not solved (small skill bonus).",
    "Each student favours some topics (Dirichlet-distributed topic preferences).",
    "Attempt times follow exponential gaps (mean 0.6 days) from a random start in early 2025.",
]
LIMITATIONS = [
    "Not real student data; produced by a simulation designed by the project author.",
    "Results on it cannot establish real-world predictive performance.",
    "Real learners differ in ways this simulation does not model (fatigue, copying, hints, etc.).",
    "Uses only topic and difficulty level as question descriptors, matching what the app knows.",
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_attempts(n_students: int = 600, seed: int = 42) -> pd.DataFrame:
    """Simulate attempts. Same seed and size always give the same table."""
    if n_students < 1:
        raise ValueError("n_students must be at least 1.")
    rng = np.random.default_rng(seed)
    n_topics = len(TOPICS)

    topic_offset = rng.normal(0.0, 0.5, n_topics)
    pool: dict[tuple[int, str], list[tuple[str, float]]] = {}
    for ti in range(n_topics):
        for d in DIFFICULTIES:
            pool[(ti, d)] = [
                (f"SYN-T{ti:02d}-{d[0].upper()}-{k}", float(rng.normal(0.0, 0.4)))
                for k in range(QUESTIONS_PER_CELL)
            ]
    qinfo = {qid: (ti, d, off) for (ti, d), items in pool.items() for qid, off in items}

    rows: list[dict] = []
    attempt_id = 1
    for user_id in range(1, n_students + 1):
        theta = rng.normal(0.0, 1.0)
        skill = theta + rng.normal(0.0, 0.6, n_topics)
        learn_rate = rng.uniform(0.02, 0.10)
        focus = rng.dirichlet(np.full(n_topics, 2.0))
        n_attempts = int(rng.integers(25, 121))
        clock = START_DATE + pd.Timedelta(days=float(rng.uniform(0, 60)))

        topic_counts = np.zeros(n_topics, dtype=int)
        outcomes: list[int] = []
        attempted: set[str] = set()
        solved: set[str] = set()
        failed_unsolved: list[str] = []

        for _ in range(n_attempts):
            recent = outcomes[-5:]
            recent_rate = (sum(recent) + 1.0) / (len(recent) + 2.0)

            if failed_unsolved and rng.random() < 0.2:
                qid = failed_unsolved[int(rng.integers(len(failed_unsolved)))]
                ti, d, offset = qinfo[qid]
            else:
                ti = int(rng.choice(n_topics, p=focus))
                weights = np.array([1.2 - recent_rate, 0.8, max(0.05, recent_rate - 0.2)])
                d = DIFFICULTIES[int(rng.choice(3, p=weights / weights.sum()))]
                candidates = [q for q in pool[(ti, d)] if q[0] not in solved] or pool[(ti, d)]
                qid, offset = candidates[int(rng.integers(len(candidates)))]

            retry_bonus = 0.3 if qid in attempted else 0.0
            effective_skill = skill[ti] + learn_rate * min(int(topic_counts[ti]), 15) + retry_bonus
            hardness = DIFFICULTY_HARDNESS[d] + topic_offset[ti] + offset
            p = _sigmoid(SLOPE * (effective_skill - hardness))
            correct = int(rng.random() < p)

            rows.append(
                {
                    "attempt_id": attempt_id,
                    "user_id": user_id,
                    "question_id": qid,
                    "topic": TOPICS[ti],
                    "difficulty": d,
                    "attempted_at": clock.floor("s"),
                    "is_correct": correct,
                    "sim_true_prob": round(float(p), 4),  # DIAGNOSTIC ONLY, never a feature
                }
            )
            attempt_id += 1

            outcomes.append(correct)
            topic_counts[ti] += 1
            attempted.add(qid)
            if correct:
                solved.add(qid)
                if qid in failed_unsolved:
                    failed_unsolved.remove(qid)
            elif qid not in solved and qid not in failed_unsolved:
                failed_unsolved.append(qid)
            clock = clock + pd.Timedelta(days=float(rng.exponential(0.6)))

    df = pd.DataFrame(rows)
    validate_attempts(df)  # fail loudly if the generator ever produces an invalid table
    return df


def save_dataset(df: pd.DataFrame, directory: Path, seed: int) -> Path:
    """Write the CSV and a metadata JSON next to it. Returns the CSV path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "synthetic_attempts.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "dataset_name": DATASET_NAME,
        "dataset_type": "synthetic",
        "seed": seed,
        "n_rows": int(len(df)),
        "n_students": int(df["user_id"].nunique()),
        "overall_correct_rate": round(float(df["is_correct"].mean()), 4),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns": list(df.columns),
        "diagnostic_columns_not_features": ["sim_true_prob"],
        "assumptions": ASSUMPTIONS,
        "limitations": LIMITATIONS,
    }
    (directory / "synthetic_attempts.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic attempts dataset.")
    parser.add_argument("--students", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_attempts(args.students, args.seed)
    path = save_dataset(df, get_settings().ml_data_dir, args.seed)

    print(f"SYNTHETIC dataset written to {path}")
    print(f"Rows: {len(df)}  Students: {df['user_id'].nunique()}  Seed: {args.seed}")
    print(f"Overall correct rate: {df['is_correct'].mean():.3f}")
    for d in DIFFICULTIES:
        sub = df[df["difficulty"] == d]
        print(f"  {d:<6} correct rate {sub['is_correct'].mean():.3f}  (n={len(sub)})")
    print("Reminder: this is simulated data, not real students.")


if __name__ == "__main__":
    main()