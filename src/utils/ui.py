"""Shared UI helpers: CSS, page headers and badges."""

import streamlit as st

_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }
footer { visibility: hidden; }
.sb-badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; margin-right: 6px;
}
.sb-easy   { background: #DCFCE7; color: #166534; }
.sb-medium { background: #FEF3C7; color: #92400E; }
.sb-hard   { background: #FEE2E2; color: #991B1B; }
.sb-topic  { background: #DBEAFE; color: #1E3A8A; }
.sb-tag {
    display: inline-block; padding: 1px 8px; border-radius: 6px; font-size: 0.75rem;
    background: #F1F5F9; color: #475569; margin-right: 4px; border: 1px solid #E2E8F0;
}
.sb-brand { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0; }
.sb-tagline { color: #475569; margin-top: 0; }
</style>
"""

_BADGE_CLASS = {"easy": "sb-easy", "medium": "sb-medium", "hard": "sb-hard"}


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def brand_header() -> None:
    st.markdown('<p class="sb-brand">SkillBridge AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sb-tagline">Adaptive coding assessment and personalized learning</p>',
        unsafe_allow_html=True,
    )


def difficulty_badge(difficulty: str) -> str:
    css = _BADGE_CLASS.get(difficulty, "sb-topic")
    return f'<span class="sb-badge {css}">{difficulty.capitalize()}</span>'


def topic_badge(topic: str) -> str:
    return f'<span class="sb-badge sb-topic">{topic}</span>'


def tag_chips(tags: list[str]) -> str:
    return "".join(f'<span class="sb-tag">{t}</span>' for t in tags)