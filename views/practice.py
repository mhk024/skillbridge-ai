"""Practice Questions: browse and read problems."""

import streamlit as st

from src.assessment import question_service as qs
from src.database.connection import session_scope
from src.utils.ui import difficulty_badge, page_header, tag_chips, topic_badge

page_header(
    "Practice Questions",
    "Browse the question bank and read problems. Submitting solutions arrives with the Coding Assessment.",
)

with session_scope() as session:
    topics = qs.list_topics(session)

col_topic, col_diff = st.columns(2)
topic_choice = col_topic.selectbox("Topic", ["All topics"] + topics)
diff_choice = col_diff.selectbox("Difficulty", ["All difficulties", "easy", "medium", "hard"])

with session_scope() as session:
    rows = qs.list_questions(
        session,
        topic=None if topic_choice == "All topics" else topic_choice,
        difficulty=None if diff_choice == "All difficulties" else diff_choice,
    )
    questions = [qs.public_view(q) for q in rows]  # converted inside the session

if not questions:
    st.info("No questions match these filters. Try a different topic or difficulty.")
    st.stop()

st.caption(f"{len(questions)} question(s) found")

by_id = {q["id"]: q for q in questions}
selected_id = st.selectbox(
    "Choose a question",
    list(by_id),
    format_func=lambda qid: f"{qid} · {by_id[qid]['title']} ({by_id[qid]['difficulty']})",
)
q = by_id[selected_id]

with st.container(border=True):
    st.subheader(q["title"])
    st.markdown(topic_badge(q["topic"]) + difficulty_badge(q["difficulty"]), unsafe_allow_html=True)
    st.write(q["statement"])

    st.markdown("**Input format**")
    st.write(q["input_format"])
    st.markdown("**Output format**")
    st.write(q["output_format"])
    st.markdown("**Constraints**")
    st.write(q["constraints"])

    st.markdown("**Examples**")
    for i, ex in enumerate(q["examples"], start=1):
        left, right = st.columns(2)
        left.caption(f"Example {i}: input")
        left.code(ex["input"].rstrip("\n"), language=None)
        right.caption(f"Example {i}: output")
        right.code(ex["output"], language=None)

    st.markdown("**What you will practise**")
    for objective in q["learning_objectives"]:
        st.write(f"- {objective}")

    st.markdown(tag_chips(q["tags"]), unsafe_allow_html=True)