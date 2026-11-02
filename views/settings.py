"""Profile and Settings."""

import streamlit as st

from src.auth.session import current_user_id
from src.database import operations as ops
from src.database.connection import session_scope
from src.utils.config import get_settings
from src.utils.ui import page_header

page_header("Profile and Settings", "Your account and the application's configuration status.")

settings = get_settings()

with session_scope() as session:
    user = ops.get_user_by_id(session, current_user_id())
    profile = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name or "Not set",
        "created": user.created_at.strftime("%d %b %Y"),
    }
    schema_version = ops.get_schema_version(session)

st.subheader("Profile")
with st.container(border=True):
    c1, c2 = st.columns(2)
    c1.markdown(f"**Username**  \n{profile['username']}")
    c1.markdown(f"**Full name**  \n{profile['full_name']}")
    c2.markdown(f"**Email**  \n{profile['email']}")
    c2.markdown(f"**Member since**  \n{profile['created']}")
st.caption("Profile editing and password change are not implemented yet.")

st.subheader("System status")
with st.container(border=True):
    if settings.llm_configured:
        st.success(f"AI assistant: API key configured (provider: {settings.llm_provider}).")
    else:
        st.warning(
            "AI assistant: no API key configured. Core features work; AI explanations "
            "will use built-in fallbacks once that feature is added."
        )
    if settings.enable_code_execution:
        st.info("Code execution: enabled (local mode, see documentation for the security boundary).")
    else:
        st.info("Code execution: disabled by configuration.")

    models = list(settings.models_dir.glob("*.joblib")) if settings.models_dir.exists() else []
    if models:
        st.success(f"ML model artifacts found: {len(models)} file(s).")
    else:
        st.warning("ML model: not trained yet. Training is implemented in a later phase.")

st.subheader("About")
st.write(
    f"SkillBridge AI, version 0.1.0 (development). Environment: {settings.app_env}. "
    f"Database schema version: {schema_version}."
)