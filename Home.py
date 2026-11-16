"""Expresso Churn Intelligence Hub: entrypoint and router.

Run with:  streamlit run Home.py
"""

import streamlit as st

st.set_page_config(
    page_title="Expresso Churn Intelligence Hub",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="auto",
)

from app import navigation  # noqa: E402
from app.core import state  # noqa: E402
from app.ui import layout  # noqa: E402

state.init()
layout.inject_css()

if not layout.password_gate():
    st.stop()

page = navigation.build(st.session_state.get("role", "all"))
layout.handle_goto()
layout.render_sidebar()
layout.render_present_exit()
layout.render_tour_card()
layout.run_page(page)
layout.render_footer()
