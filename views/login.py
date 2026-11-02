"""Login and registration page."""

import logging

import streamlit as st

from src.auth import service as auth
from src.auth.session import login_user
from src.database.connection import session_scope
from src.utils.ui import brand_header

logger = logging.getLogger("skillbridge")

brand_header()
st.write("")

_, centre, _ = st.columns([1, 2, 1])

with centre:
    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            try:
                with session_scope() as session:
                    user = auth.authenticate(session, identifier, password)
                login_user(user.id)
                st.rerun()
            except auth.AuthError as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("Login failed unexpectedly")
                st.error("Something went wrong. Please try again.")

    with register_tab:
        with st.form("register_form"):
            full_name = st.text_input("Full name (optional)")
            new_username = st.text_input("Username", help="3-30 characters: letters, numbers, underscores.")
            new_email = st.text_input("Email")
            new_password = st.text_input(
                "Password", type="password", help="At least 8 characters with a letter and a number."
            )
            confirm = st.text_input("Confirm password", type="password")
            created = st.form_submit_button("Create account", type="primary")
        if created:
            if new_password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    with session_scope() as session:
                        user = auth.register_user(
                            session, new_username, new_email, new_password, full_name
                        )
                    login_user(user.id)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
                except Exception:
                    logger.exception("Registration failed unexpectedly")
                    st.error("Something went wrong. Please try again.")