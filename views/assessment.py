"""Coding Assessment: write a solution, test it, submit it for judging."""

import logging

import pandas as pd
import streamlit as st

from src.assessment import question_service as qs
from src.assessment.evaluation_service import evaluate_code
from src.assessment.submission_service import record_evaluation
from src.auth.session import current_user_id
from src.database import operations as ops
from src.database.connection import session_scope
from src.utils.config import get_settings
from src.utils.helpers import utc_to_local_str
from src.utils.ui import difficulty_badge, page_header, tag_chips, topic_badge
from src.ml.predict import (
    TRAIN_COMMAND,
    ModelUnavailableError,
    load_model,
    model_file_mtime,
    predict_for_next_attempt,
)

logger = logging.getLogger("skillbridge")


@st.cache_resource(show_spinner=False)
def _cached_model(file_stamp: float):
    """Load the model once per model-file version (the stamp is the cache key)."""
    return load_model()

STARTER_CODE = (
    "# Read input with input() and print your answer with print().\n"
    "# Example:  n = int(input())\n"
)
STATUS_LABELS = {
    "passed": "Passed",
    "wrong_answer": "Wrong answer",
    "runtime_error": "Runtime error",
    "timeout": "Time limit exceeded",
    "output_limit": "Output limit exceeded",
    "internal_error": "Runner error",
}

st.markdown(
    "<style>div[data-testid='stTextArea'] textarea "
    "{ font-family: Consolas, 'Courier New', monospace; font-size: 0.9rem; }</style>",
    unsafe_allow_html=True,
)


def render_result(res: dict) -> None:
    ev, mode, saved = res["evaluation"], res["mode"], res["saved"]

    if ev.status in ("invalid", "disabled"):
        st.warning(ev.message)
        return
    if ev.status == "syntax_error":
        st.error(f"Syntax error: {ev.message}")
        if saved:
            st.caption(f"Saved as attempt {saved['attempt']} (a syntax error counts as a failed attempt).")
        return

    if ev.is_correct:
        st.success(f"All {ev.total} tests passed.")
    else:
        st.error(f"{ev.passed} of {ev.total} tests passed.")

    if mode == "sample":
        st.caption(
            "Sample run: only the visible examples were run. This is a quick check, not proof "
            "that your solution is correct, and it was not saved. Use **Submit for judging** "
            "to run the hidden tests too."
        )
    elif saved:
        st.caption(f"Saved as attempt {saved['attempt']} for this question.")
    if ev.avg_duration_ms is not None:
        st.caption(f"Average run time per test: {ev.avg_duration_ms:.0f} ms (includes Python start-up).")
    if ev.memory_limit_enforced is False:
        st.warning("The memory limit could not be enforced on this system.")

    for o in ev.outcomes:
        label = STATUS_LABELS.get(o.status, o.status)
        title = f"Test {o.index}{' (hidden)' if o.hidden else ''}: {label}"
        if o.hidden:
            st.markdown(f"- {title}")
            continue
        with st.expander(title, expanded=(o.status != "passed")):
            c1, c2, c3 = st.columns(3)
            c1.caption("Input")
            c1.code(o.input_text or "(empty)", language=None)
            c2.caption("Expected output")
            c2.code(o.expected or "(empty)", language=None)
            c3.caption("Your output")
            c3.code(o.actual if o.actual else "(none)", language=None)
            if o.error:
                st.code(o.error, language=None)


page_header(
    "Coding Assessment",
    "Write Python solutions. Run the sample tests to check, then submit for full judging.",
)

settings = get_settings()
user_id = current_user_id()

if not settings.enable_code_execution:
    st.warning(
        "Code execution is disabled in this deployment, so solutions cannot be run. "
        "You can still read problems in Practice Questions."
    )
    st.stop()

# ---------------------------------------------------------- find the active assessment
active_id = st.session_state.get("active_assessment_id")
with session_scope() as session:
    active = None
    if active_id is not None:
        active = ops.get_assessment_for_user(session, active_id, user_id)
        if active is not None and active.completed_at is not None:
            active = None
    if active is None:
        active = ops.get_open_assessment(session, user_id)  # resume an unfinished one
    info = (
        None
        if active is None
        else {"id": active.id, "topic_filter": active.topic_filter}
    )
    topics = qs.list_topics(session)

# ------------------------------------------------------------------ start a new one
if info is None:
    st.session_state.pop("active_assessment_id", None)

    summary = st.session_state.get("last_summary")
    if summary:
        if summary["attempts"] == 0:
            st.info("Your last assessment was closed with no attempts.")
        else:
            st.success(
                f"Assessment finished: you solved {summary['solved']} of {summary['attempted']} "
                f"attempted question(s) (score {summary['score']}%) in {summary['attempts']} attempt(s)."
            )

    with st.container(border=True):
        st.subheader("Start a new assessment")
        topic_choice = st.selectbox("Topic", ["Mixed (all topics)"] + topics)
        st.caption(
            "Submitted attempts are saved to your history. Sample runs are for practice and are not saved."
        )
        if st.button("Start assessment", type="primary"):
            with session_scope() as session:
                created = ops.create_assessment(
                    session,
                    user_id=user_id,
                    mode="judge",
                    topic_filter=None if topic_choice.startswith("Mixed") else topic_choice,
                )
                new_id = created.id
            st.session_state["active_assessment_id"] = new_id
            st.session_state.pop("last_summary", None)
            st.rerun()
    st.stop()

# ----------------------------------------------------------- active assessment header
st.session_state["active_assessment_id"] = info["id"]
assessment_id = info["id"]

head_left, head_right = st.columns([3, 2])
head_left.markdown(f"**Assessment #{assessment_id}** · {info['topic_filter'] or 'Mixed topics'}")
difficulty = head_right.selectbox("Difficulty", ["All", "easy", "medium", "hard"])

with session_scope() as session:
    rows = qs.list_questions(
        session,
        topic=info["topic_filter"],
        difficulty=None if difficulty == "All" else difficulty,
    )
    meta = {q.id: {"title": q.title, "difficulty": q.difficulty} for q in rows}
    solved_ids = ops.get_solved_question_ids(session, user_id)
    attempted_ids = ops.get_attempted_question_ids(session, user_id)
    progress = ops.get_assessment_progress(session, assessment_id, user_id)

st.caption(
    f"This assessment: {progress['attempts']} attempt(s) · {progress['solved']} solved "
    f"of {progress['attempted']} attempted question(s)"
)


def question_label(qid: str) -> str:
    status = "Solved" if qid in solved_ids else ("Tried" if qid in attempted_ids else "New")
    return f"[{status}] {qid} · {meta[qid]['title']} ({meta[qid]['difficulty']})"


if not meta:
    st.info("No questions match this topic and difficulty. Try another difficulty.")
else:
    qid = st.selectbox("Question", list(meta), format_func=question_label, key=f"qsel_{assessment_id}")

    with session_scope() as session:
        question = qs.get_question(session, qid)
        view = qs.public_view(question)
        tests = [dict(t) for t in question.test_cases]  # full tests stay on the server

    with st.container(border=True):
        st.subheader(view["title"])
        st.markdown(topic_badge(view["topic"]) + difficulty_badge(view["difficulty"]), unsafe_allow_html=True)
        st.write(view["statement"])
        st.markdown(f"**Input:** {view['input_format']}")
        st.markdown(f"**Output:** {view['output_format']}")
        st.markdown(f"**Constraints:** {view['constraints']}")
        with st.expander("Examples", expanded=True):
            for i, ex in enumerate(view["examples"], start=1):
                left, right = st.columns(2)
                left.caption(f"Example {i}: input")
                left.code(ex["input"].rstrip("\n") or "(empty)", language=None)
                right.caption(f"Example {i}: output")
                right.code(ex["output"] or "(empty)", language=None)
        st.markdown(tag_chips(view["tags"]), unsafe_allow_html=True)
        # ---- ML estimate: made BEFORE this attempt, from your earlier saved attempts only
    predicted_prob = None
    model_version = None
    stamp = model_file_mtime()
    if stamp is None:
        st.caption(f"ML estimate unavailable: no trained model yet. Train one with `{TRAIN_COMMAND}`.")
    else:
        try:
            model = _cached_model(stamp)
            with session_scope() as session:
                predicted_prob = predict_for_next_attempt(
                    session, model, user_id=user_id, question_id=qid,
                    topic=view["topic"], difficulty=view["difficulty"],
                )
            model_version = model.version
            st.info(f"Estimated chance you solve this on your next attempt: **{predicted_prob:.0%}**")
            if model.dataset_type == "synthetic":
                st.caption(
                    "This model was trained on SYNTHETIC (simulated) data, so treat the number "
                    f"as a demonstration, not a measurement of you. Model: {model.version}"
                )
            else:
                st.caption(f"Model: {model.version}")
        except ModelUnavailableError as exc:
            st.caption(f"ML estimate unavailable: {exc}")
        except Exception:
            logger.exception("Prediction failed")
            st.caption("ML estimate unavailable right now.")

    # ------------------------------------------------------------------ the editor
    code_key = f"code_{assessment_id}_{qid}"
    if code_key not in st.session_state:
        st.session_state[code_key] = STARTER_CODE
    code = st.text_area("Your Python solution", key=code_key, height=300)
    st.caption(
        "Tip: the Tab key does not indent here, so type spaces. Click outside the editor "
        "(or press Ctrl+Enter) before running if the last change does not seem to register."
    )

    b1, b2, _ = st.columns([1, 1, 2])
    run_clicked = b1.button("Run sample tests (not saved)")
    submit_clicked = b2.button("Submit for judging", type="primary")

    result_key = f"result_{assessment_id}_{qid}"
    if run_clicked or submit_clicked:
        mode = "judge" if submit_clicked else "sample"
        with st.spinner("Running your code..."):
            evaluation = evaluate_code(code, tests, mode=mode)
        saved = None
        save_failed = False
        if submit_clicked and evaluation.counts_as_attempt:
            try:
                with session_scope() as session:
                    sub = record_evaluation(
                        session,
                        user_id=user_id,
                        assessment_id=assessment_id,
                        question_id=qid,
                        code=code,
                        evaluation=evaluation,
                        predicted_prob=predicted_prob,
                        model_version=model_version,
                    )
                    saved = {"attempt": sub.attempt_number, "correct": sub.is_correct}
            except Exception:
                logger.exception("Could not save submission")
                save_failed = True
        st.session_state[result_key] = {"evaluation": evaluation, "mode": mode, "saved": saved}
        if save_failed:
            st.error("Your code ran, but the result could not be saved. Please submit again.")
        elif saved is not None:
            st.rerun()  # refresh progress counters and Solved/Tried labels

    result = st.session_state.get(result_key)
    if result:
        render_result(result)

    with session_scope() as session:
        history = [
            {
                "Attempt": s.attempt_number,
                "When": utc_to_local_str(s.submitted_at),
                "Tests passed": f"{s.passed_tests}/{s.total_tests}",
                "Result": "Correct" if s.is_correct else "Not correct",
            }
            for s in ops.get_question_submissions(session, user_id, qid)
        ]
    with st.expander(f"Your previous attempts on this question ({len(history)})"):
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
        else:
            st.caption("No saved attempts yet.")

# ------------------------------------------------------------------------- finish
st.divider()
if st.button("Finish assessment"):
    with session_scope() as session:
        final_progress = ops.get_assessment_progress(session, assessment_id, user_id)
        closed = ops.complete_assessment(session, assessment_id, user_id)
        score = closed.score
    st.session_state["last_summary"] = {**final_progress, "score": score}
    st.session_state.pop("active_assessment_id", None)
    st.rerun()