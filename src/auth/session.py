"""Streamlit session helpers for the logged-in user."""

import streamlit as st

_USER_ID_KEY = "auth_user_id"


def login_user(user_id: int) -> None:
    st.session_state[_USER_ID_KEY] = user_id


def logout_user() -> None:
    st.session_state.clear()


def current_user_id() -> int | None:
    return st.session_state.get(_USER_ID_KEY)


def is_logged_in() -> bool:
    return current_user_id() is not None