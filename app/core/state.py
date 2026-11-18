"""Session-state helpers shared across pages."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

DEFAULTS = {
    "lang": "en",
    "role": "all",
    "present_mode": False,
    "tour_step": None,
    "auth_ok": False,
}


def init() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Scored batch (shared by Batch Scoring, Retention Planner, Copilot, ...)
# ---------------------------------------------------------------------------
def set_batch(scored: pd.DataFrame, name: str, synthetic: bool, raw: pd.DataFrame | None = None,
              drift: dict | None = None) -> None:
    st.session_state["batch"] = {
        "df": scored,
        "raw": raw,
        "name": name,
        "synthetic": synthetic,
        "rows": int(len(scored)),
        "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "drift": drift,
        "id": f"{name}-{len(scored)}-{datetime.now(timezone.utc).timestamp():.0f}",
    }


def get_batch() -> dict | None:
    return st.session_state.get("batch")


def clear_batch() -> None:
    st.session_state.pop("batch", None)


# ---------------------------------------------------------------------------
# Customer followed on the Customer Journey page (also the Copilot's what-if base)
# ---------------------------------------------------------------------------
def set_last_customer(raw_row: dict) -> None:
    st.session_state["last_customer"] = dict(raw_row)


def get_last_customer() -> dict | None:
    return st.session_state.get("last_customer")
