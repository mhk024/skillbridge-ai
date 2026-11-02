"""SkillBridge AI: application entrypoint.

Run with:  streamlit run app.py
"""

import logging

import streamlit as st

from src.assessment.question_service import load_question_bank, sync_questions
from src.auth.session import current_user_id, logout_user
from src.database import operations as ops
from src.database.connection import get_engine, init_db, session_scope
from src.utils.config import get_settings
from src.utils.ui import inject_css

logger = logging.getLogger("skillbridge")

st.set_page_config(page_title="SkillBridge AI", page_icon="🎓", layout="wide")
inject_css()


@st.cache_resource(show_spinner="Preparing database...")
def bootstrap() -> dict:
    """Create tables and load the question bank. Runs once per server start."""
    settings = get_settings()
    engine = get_engine()
    init_db(engine)
    questions = load_question_bank(settings.questions_dir)
    with session_scope(engine) as session:
        return sync_questions(session, questions)


try:
    bootstrap()
except Exception:
    logger.exception("Startup failed")
    st.error(
        "SkillBridge AI could not start. Check that the database folder is writable and "
        "that data/questions/*.json is valid (run `python -m scripts.validate_questions`)."
    )
    st.stop()

# Make sure the logged-in user still exists (e.g. the database was recreated).
user_id = current_user_id()
username = None
if user_id is not None:
    with session_scope() as session:
        user = ops.get_user_by_id(session, user_id)
        username = user.username if user else None
    if username is None:
        logout_user()
        st.rerun()

if username is None:
    login_page = st.Page("views/login.py", title="Login / Register", icon=":material/login:")
    navigation = st.navigation([login_page], position="hidden")
else:
    navigation = st.navigation(
        [
            st.Page(
                "views/practice.py",
                title="Practice Questions",
                icon=":material/menu_book:",
                default=True,
            ),
            st.Page(
                "views/settings.py",
                title="Profile and Settings",
                icon=":material/settings:",
            ),
        ]
    )
    with st.sidebar:
        st.divider()
        st.caption(f"Signed in as **{username}**")
        if st.button("Log out", icon=":material/logout:"):
            logout_user()
            st.rerun()

navigation.run()