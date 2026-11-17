"""Adaptive question recommendations.

STRATEGY (transparent and rule-based; this is NOT reinforcement learning):
  1. Per topic, summarise RECENT performance (last TOPIC_WINDOW attempts, smoothed accuracy)
     and which difficulty levels the student has solved.
  2. Classify each topic: new / struggling / developing / strong.
  3. Choose a target difficulty by rule:
       new or struggling -> easy (foundation)
       developing        -> highest level solved so far (consolidate)
       strong            -> one level above the highest solved (step up)
  4. Rank topics by need. Weak topics come first; a topic whose prerequisite is struggling is
     demoted, and that prerequisite is boosted if the student has started the dependent topic.
     Ties go to the order of TOPICS (basics first).
  5. The ML estimate SUPPORTS the rules, it does not control them: it can nudge the level one
     step down/up when its average estimate is extreme, and it ranks questions at the chosen
     level, preferring a "productive struggle" zone (45%-75% chance of success).
  6. Solved questions and caller-excluded questions are never offered.
Every recommendation carries a plain-language reason.

The prerequisite map and all thresholds are the project author's judgement, not derived from data.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.assessment.question_service import TOPICS
from src.ml.predict import predict_proba
from src.ml.preprocessing import (
    DIFFICULTY_LEVEL,
    PRIOR_RATE,
    PRIOR_STRENGTH,
    REQUIRED_ATTEMPT_COLUMNS,
    features_for_next_attempt,
    validate_attempts,
)

TOPIC_WINDOW = 8
MIN_ATTEMPTS = 3
STRUGGLING_BELOW = 0.40
STRONG_AT_LEAST = 0.65
ZONE_LOW, ZONE_HIGH = 0.45, 0.75
ML_STEP_DOWN_BELOW = 0.30
ML_STEP_UP_ABOVE = 0.85

LEVEL_OF = DIFFICULTY_LEVEL
LEVEL_NAME = {1: "easy", 2: "medium", 3: "hard"}

PREREQUISITES = {
    "Hashing": ("Arrays",),
    "Searching": ("Arrays",),
    "Sorting": ("Arrays",),
    "Two Pointers": ("Arrays",),
    "Sliding Window": ("Arrays", "Strings"),
    "Binary Search": ("Arrays", "Searching"),
    "Stacks and Queues": ("Arrays",),
    "Trees": ("Recursion", "Stacks and Queues"),
    "Graphs": ("Trees",),
    "Dynamic Programming": ("Recursion", "Arrays"),
}

_BASE_STRATEGY = {
    "new": "explore",
    "struggling": "foundation",
    "developing": "consolidate",
    "strong": "step_up",
}


@dataclass(frozen=True)
class TopicStatus:
    topic: str
    attempts: int  # all-time attempts in this topic
    accuracy: float  # smoothed accuracy over the last TOPIC_WINDOW attempts
    solved: dict  # level (1-3) -> number of distinct solved questions
    highest_solved: int  # 0 if nothing solved yet
    state: str  # new | struggling | developing | strong


@dataclass(frozen=True)
class Recommendation:
    question_id: str
    title: str
    topic: str
    difficulty: str
    strategy: str  # explore | foundation | consolidate | step_up | ml_step_down | ml_step_up
    reason: str
    predicted_prob: float | None


# ---------------------------------------------------------------------- topic status
def _clean(attempts: pd.DataFrame | None) -> pd.DataFrame:
    if attempts is None or len(attempts) == 0:
        return pd.DataFrame(columns=REQUIRED_ATTEMPT_COLUMNS)
    df = validate_attempts(attempts)
    return df.sort_values(["attempted_at", "attempt_id"], kind="mergesort").reset_index(drop=True)


def _statuses_from_df(df: pd.DataFrame) -> dict[str, TopicStatus]:
    out = {}
    for topic in TOPICS:
        t = df[df["topic"] == topic]
        recent = t.tail(TOPIC_WINDOW)
        accuracy = (float(recent["is_correct"].sum()) + PRIOR_RATE * PRIOR_STRENGTH) / (
            len(recent) + PRIOR_STRENGTH
        )
        solved_rows = t[t["is_correct"] == 1].drop_duplicates("question_id")
        solved = {
            level: int((solved_rows["difficulty"] == name).sum())
            for name, level in LEVEL_OF.items()
        }
        highest = max((lvl for lvl, count in solved.items() if count > 0), default=0)
        if len(t) == 0:
            state = "new"
        elif len(recent) >= MIN_ATTEMPTS and accuracy < STRUGGLING_BELOW:
            state = "struggling"
        elif len(recent) >= MIN_ATTEMPTS and accuracy >= STRONG_AT_LEAST:
            state = "strong"
        else:
            state = "developing"
        out[topic] = TopicStatus(topic, len(t), accuracy, solved, highest, state)
    return out


def topic_statuses(attempts: pd.DataFrame | None) -> dict[str, TopicStatus]:
    """Per-topic summary of one student's history (None or empty = a new student)."""
    return _statuses_from_df(_clean(attempts))


# -------------------------------------------------------------------- the ML helper
def make_predictor(model, history: pd.DataFrame | None, user_id: int) -> Callable[[dict], float | None]:
    """question dict -> estimated probability of solving it next (None if anything fails)."""

    def predict(question: dict) -> float | None:
        try:
            features = features_for_next_attempt(
                history,
                user_id=user_id,
                question_id=question["id"],
                topic=question["topic"],
                difficulty=question["difficulty"],
            )
            return float(predict_proba(model, features)[0])
        except Exception:
            return None

    return predict


# ------------------------------------------------------------------ selection rules
def _target_level(status: TopicStatus) -> int:
    if status.state in ("new", "struggling"):
        return 1
    if status.state == "strong":
        return min(3, max(status.highest_solved, 1) + 1)
    return max(status.highest_solved, 1)  # developing


def _levels_to_try(target: int) -> list[int]:
    order = [target, target - 1, target + 1, target - 2, target + 2]
    seen, levels = set(), []
    for level in order:
        if 1 <= level <= 3 and level not in seen:
            levels.append(level)
            seen.add(level)
    return levels


def _zone_score(p: float | None) -> float:
    if p is None:
        return 0.5  # neutral without a model estimate
    if ZONE_LOW <= p <= ZONE_HIGH:
        return 1.0
    gap = ZONE_LOW - p if p < ZONE_LOW else p - ZONE_HIGH
    return max(0.0, 1.0 - gap / 0.4)


def _ranked_for_topic(status, unsolved, attempted_ids, predict) -> list[dict]:
    by_level = {1: [], 2: [], 3: []}
    for q in unsolved:
        by_level[LEVEL_OF[q["difficulty"]]].append(q)
    for items in by_level.values():
        items.sort(key=lambda q: q["id"])

    target = _target_level(status)
    level = next((lv for lv in _levels_to_try(target) if by_level[lv]), None)
    if level is None:
        return []

    cache: dict[str, float | None] = {}

    def prob(q):
        if predict is None:
            return None
        if q["id"] not in cache:
            try:
                value = predict(q)
            except Exception:
                value = None
            cache[q["id"]] = None if value is None else float(value)
        return cache[q["id"]]

    strategy = _BASE_STRATEGY[status.state]
    ml_avg = None
    if predict is not None and level == target:
        estimates = [p for p in (prob(q) for q in by_level[level][:3]) if p is not None]
        if estimates:
            ml_avg = sum(estimates) / len(estimates)
            if ml_avg < ML_STEP_DOWN_BELOW:
                lower = next((lv for lv in range(level - 1, 0, -1) if by_level[lv]), None)
                if lower is not None:
                    level, strategy = lower, "ml_step_down"
            elif ml_avg > ML_STEP_UP_ABOVE and status.state != "struggling":
                higher = next((lv for lv in range(level + 1, 4) if by_level[lv]), None)
                if higher is not None:
                    level, strategy = higher, "ml_step_up"

    ranked = sorted(
        by_level[level],
        key=lambda q: (-_zone_score(prob(q)), 1 if q["id"] in attempted_ids else 0, q["id"]),
    )
    return [
        {"q": q, "p": prob(q), "strategy": strategy, "level": level, "target": target, "ml_avg": ml_avg}
        for q in ranked
    ]


def _reason(status: TopicStatus, item: dict, attempted_ids: set, foundation_for: dict) -> str:
    q, p, strategy = item["q"], item["p"], item["strategy"]
    topic, diff = status.topic, q["difficulty"]
    acc = f"{status.accuracy:.0%}"
    recent_n = min(status.attempts, TOPIC_WINDOW)

    if strategy == "explore":
        text = f"You have not tried {topic} yet, so this is a good place to start ({diff})."
    elif strategy == "foundation":
        text = (
            f"Your recent accuracy in {topic} is {acc} over your last {recent_n} attempt(s), "
            "so here is a foundational question to build the basics."
        )
    elif strategy == "consolidate":
        if status.highest_solved == 0:
            text = (
                f"You have not solved a {topic} question yet and your recent accuracy is {acc}, "
                f"so this {diff} question builds the basics."
            )
        else:
            text = (
                f"You have solved {LEVEL_NAME[status.highest_solved]} {topic} questions and your "
                f"recent accuracy is {acc}; practising at this level before moving up."
            )
    elif strategy == "step_up":
        text = (
            f"Your recent accuracy in {topic} is {acc} and you have solved "
            f"{LEVEL_NAME[status.highest_solved]} questions here, so this {diff} question "
            "is the next step up."
        )
    elif strategy == "ml_step_down":
        text = (
            f"The model estimates only about {item['ml_avg']:.0%} chance on "
            f"{LEVEL_NAME[item['target']]} {topic} questions right now, so here is an easier {diff} one."
        )
    else:  # ml_step_up
        text = (
            f"The model estimates about {item['ml_avg']:.0%} chance on "
            f"{LEVEL_NAME[item['target']]} {topic} questions, so here is a harder {diff} one."
        )

    if item["level"] != item["target"] and not strategy.startswith("ml_"):
        text += (
            f" (No unsolved {LEVEL_NAME[item['target']]} question is left in {topic}, "
            f"so this {diff} one is the closest match.)"
        )
    if topic in foundation_for:
        text += f" {topic} is also a foundation for {', '.join(foundation_for[topic])}, which you have started."
    if q["id"] in attempted_ids:
        text += " You attempted it before but have not solved it yet."
    if p is not None:
        text += f" Model estimate of solving it: {p:.0%}."
    return text


# --------------------------------------------------------------------- main entry
def recommend(
    attempts: pd.DataFrame | None,
    questions: list[dict],
    *,
    predict: Callable[[dict], float | None] | None = None,
    excluded_ids=frozenset(),
    n: int = 5,
) -> list[Recommendation]:
    """Recommend up to n questions.

    attempts     : this student's history (same columns as the ML attempts table), or None.
    questions    : dicts with id, title, topic, difficulty.
    predict      : optional question dict -> probability; failures are tolerated (treated as None).
    excluded_ids : questions not to offer (e.g. recommended recently and still pending).
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    for q in questions:
        if not q.get("id") or q.get("topic") not in TOPICS or q.get("difficulty") not in LEVEL_OF:
            raise ValueError(f"Invalid question entry: {q.get('id')!r}")

    df = _clean(attempts)
    statuses = _statuses_from_df(df)
    attempted_ids = set(df["question_id"])
    solved_ids = set(df.loc[df["is_correct"] == 1, "question_id"])
    blocked = solved_ids | set(excluded_ids)

    foundation_for: dict[str, list[str]] = {}
    for topic, prereqs in PREREQUISITES.items():
        if statuses[topic].attempts > 0:
            for prereq in prereqs:
                if statuses[prereq].state == "struggling":
                    foundation_for.setdefault(prereq, []).append(topic)

    unsolved_by_topic: dict[str, list[dict]] = {t: [] for t in TOPICS}
    for q in questions:
        if q["id"] not in blocked:
            unsolved_by_topic[q["topic"]].append(q)

    priority: dict[str, float] = {}
    for topic in TOPICS:
        if not unsolved_by_topic[topic]:
            continue
        status = statuses[topic]
        value = 1.0 - status.accuracy
        if status.state == "struggling":
            value += 0.30
        elif status.state == "new":
            value += 0.10
        elif status.state == "strong":
            value -= 0.15
        if any(statuses[p].state == "struggling" for p in PREREQUISITES.get(topic, ())):
            value -= 0.30
        if topic in foundation_for:
            value += 0.15
        priority[topic] = value
    order = sorted(priority, key=lambda t: (-priority[t], TOPICS.index(t)))

    ranked: dict[str, list[dict]] = {}
    picks: list[dict] = []
    for topic in order:
        items = _ranked_for_topic(statuses[topic], unsolved_by_topic[topic], attempted_ids, predict)
        if not items:
            continue
        ranked[topic] = items
        picks.append(items[0])
        if len(picks) == n:
            break

    depth = 1  # second pass: further questions from the same topics, same order
    while len(picks) < n:
        added = False
        for topic in order:
            items = ranked.get(topic, [])
            if len(items) > depth and len(picks) < n:
                picks.append(items[depth])
                added = True
        if not added:
            break
        depth += 1

    return [
        Recommendation(
            question_id=it["q"]["id"],
            title=it["q"].get("title") or it["q"]["id"],
            topic=it["q"]["topic"],
            difficulty=it["q"]["difficulty"],
            strategy=it["strategy"],
            reason=_reason(statuses[it["q"]["topic"]], it, attempted_ids, foundation_for),
            predicted_prob=it["p"],
        )
        for it in picks
    ]