"""Save judged evaluation results as submissions."""

from sqlalchemy.orm import Session

from src.assessment.evaluation_service import EvaluationResult
from src.database import operations as ops
from src.database.models import Submission


def record_evaluation(
    session: Session,
    *,
    user_id: int,
    assessment_id: int,
    question_id: str,
    code: str,
    evaluation: EvaluationResult,
    predicted_prob: float | None = None,
    model_version: str | None = None,
) -> Submission:
    """Persist one judged attempt. Only full-judge results that count as attempts are saved."""
    if evaluation.mode != "judge":
        raise ValueError("Only full judge runs are saved as attempts.")
    if not evaluation.counts_as_attempt:
        raise ValueError("This evaluation does not count as an attempt.")
    return ops.record_submission(
        session,
        user_id=user_id,
        assessment_id=assessment_id,
        question_id=question_id,
        code=code,
        passed_tests=evaluation.passed,
        total_tests=evaluation.total,
        exec_time_ms=evaluation.avg_duration_ms,
        predicted_prob=predicted_prob,
        model_version=model_version,
    )