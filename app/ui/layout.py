"""App shell: CSS, sidebar, password gate, error boundary, footer, tour."""

from __future__ import annotations

import hmac
from functools import lru_cache
from pathlib import Path

import streamlit as st

from app import navigation
from app.core import artifacts, state
from app.core import fmt
from app.i18n import LANGUAGES, t
from app.ui import charts
from app.ui.components import badge

CSS_PATH = Path(__file__).resolve().parent / "style.css"

# "Cobalt & Graphite": the accent and card colours per mode (the rest of style.css is mode-neutral).
THEME_VARS = {
    "dark": ("--hub-accent:#6E9EEB;--hub-on-accent:#0E1420;--hub-accent-a18:rgba(110,158,235,.16);"
             "--hub-accent-a10:rgba(110,158,235,.09);--hub-good-text:#4cc38a;--hub-bad-text:#f08080;--hub-warn-text:#f0b43c;"
             "--hub-card-bg:#1F2226;--hub-line:#30343A;--hub-soft:#1B1D21;"),
    "light": ("--hub-accent:#1F5BB5;--hub-on-accent:#FFFFFF;--hub-accent-a18:rgba(31,91,181,.12);"
              "--hub-accent-a10:rgba(31,91,181,.06);--hub-good-text:#006300;--hub-bad-text:#b42318;--hub-warn-text:#8a5a00;"
              "--hub-card-bg:#FFFFFF;--hub-line:#D5D9E0;--hub-soft:#E9EBEF;"),
}

PRESENT_CSS = """
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
header[data-testid="stHeader"], [data-testid="stExpander"], [data-testid="stDownloadButton"],
[data-testid="stSlider"], [data-testid="stNumberInput"], [data-testid="stSelectbox"], [data-testid="stMultiSelect"],
[data-testid="stFileUploader"], [data-testid="stTextInput"], [data-testid="stCheckbox"], [data-testid="stRadio"],
[data-testid="stForm"], [data-testid="stButtonGroup"], [data-testid="stToggle"], [data-testid="stPageLink"],
[data-testid="stCaptionContainer"] { display:none !important; }
.block-container { max-width: 1500px !important; padding-top: 1rem !important; }
.hub-kpi-value { font-size: 2.6rem !important; }
.hub-kpi-label { font-size: 1rem !important; }
.hub-page-title { font-size: 2.8rem !important; }
.hub-sowhat { font-size: 1.1rem !important; }
.hub-callout .body { font-size: 1.12rem !important; }
"""


@lru_cache(maxsize=1)
def _base_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def inject_css() -> None:
    m = charts.mode()
    extra = PRESENT_CSS if st.session_state.get("present_mode") else ""
    st.html(f"<style>:root{{{THEME_VARS[m]}}}{_base_css()}{extra}</style>")


# ---------------------------------------------------------------------------
# Secrets and password gate
# ---------------------------------------------------------------------------
def get_secret(name: str, default=None):
    """Read a Streamlit secret without failing when no secrets file exists."""
    try:
        if not st.secrets.load_if_toml_exists():
            return default
        return st.secrets.get(name, default)
    except Exception:
        return default


def password_gate() -> bool:
    """Return True when the viewer may see the app.

    The gate is only active when ``APP_PASSWORD`` is set in st.secrets.
    """
    expected = get_secret("APP_PASSWORD")
    if not expected or st.session_state.get("auth_ok"):
        return True
    st.html(f'<div class="hub-eyebrow">{t("app.name")}</div>'
            f'<h1 class="hub-page-title">{t("auth.title")}</h1><p class="hub-page-sub">{t("auth.sub")}</p>')
    with st.form("auth_form", border=True):
        pw = st.text_input(t("auth.label"), type="password")
        ok = st.form_submit_button(t("auth.submit"), type="primary")
    if ok:
        if hmac.compare_digest(str(pw), str(expected)):
            st.session_state["auth_ok"] = True
            st.rerun()
        st.error(t("auth.wrong"))
    return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.html(f'<div style="display:flex;align-items:center;gap:.6rem;margin:.2rem 0 .6rem 0">'
                f'<div style="width:2.1rem;height:2.1rem;border-radius:.2rem;display:grid;place-items:center;'
                f'font-weight:700;background:var(--hub-accent);color:var(--hub-on-accent)">EX</div>'
                f'<div><div style="font-weight:700;line-height:1.1">{t("app.short_name")}</div>'
                f'<div style="font-size:.75rem;opacity:.7">{t("app.tagline")}</div></div></div>')

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(t("sidebar.language"), list(LANGUAGES), key="lang",
                         format_func=lambda k: LANGUAGES[k])
        with c2:
            role_labels = {k: t(f"sidebar.roles.{k}") for k in navigation.ROLES}
            st.selectbox(t("sidebar.role"), navigation.ROLES, key="role",
                         format_func=role_labels.__getitem__, help=t("sidebar.role_help"))

        ok, msg = artifacts.model_status()
        badges = [badge(t("sidebar.model_ok") if ok else t("sidebar.model_err"), "ok" if ok else "err")]
        batch = state.get_batch()
        if batch:
            kind = "warn" if batch["synthetic"] else "ok"
            badges.append(badge(t("sidebar.batch_loaded", rows=fmt.num(batch["rows"])), kind))
        else:
            badges.append(badge(t("sidebar.no_batch"), "muted"))
        st.html(f'<div style="display:flex;flex-wrap:wrap;gap:.35rem;margin:.2rem 0 .4rem 0">{"".join(badges)}</div>')
        if not ok:
            st.caption(msg)

        with st.popover(t("sidebar.glossary"), icon=":material/menu_book:", width="stretch"):
            for term in ("churn", "probability", "roc_auc", "pp", "segment", "reason_code", "decile",
                         "lift", "psi", "exploratory_band"):
                st.markdown(f"**{t(f'glossary.{term}.term')}**: {t(f'glossary.{term}.def')}")
        st.toggle(t("sidebar.present"), key="present_mode", help=t("sidebar.present_help"))
        if navigation.available("home"):
            if st.button(t("sidebar.tour"), icon=":material/explore:", width="stretch"):
                start_tour()


# ---------------------------------------------------------------------------
# Guided tour (session-state driven, no external JS)
# ---------------------------------------------------------------------------
TOUR = ["home", "problem", "executive", "segments", "customer_check", "what_if", "batch", "retention",
        "evaluation", "copilot", "reports"]


def tour_steps() -> list[str]:
    return [k for k in TOUR if navigation.available(k)]


def start_tour() -> None:
    st.session_state["tour_step"] = 0
    navigation.goto(tour_steps()[0])


def _tour_move(delta: int) -> None:
    steps = tour_steps()
    nxt = (st.session_state.get("tour_step") or 0) + delta
    if nxt < 0 or nxt >= len(steps):
        st.session_state["tour_step"] = None
        return
    st.session_state["tour_step"] = nxt
    st.session_state["_goto"] = steps[nxt]


def request_goto(key: str, role: str | None = None) -> None:
    """Callback-safe navigation: switch on the next run (optionally set role)."""
    if role:
        st.session_state["role"] = role
    st.session_state["_goto"] = key


def handle_goto() -> None:
    target = st.session_state.pop("_goto", None)
    if target:
        navigation.goto(target)


def render_tour_card() -> None:
    step = st.session_state.get("tour_step")
    steps = tour_steps()
    if step is None or step >= len(steps):
        return
    key = steps[step]
    st.html(f'<div class="hub-tour"><div class="n">{t("tour.step", n=step + 1, total=len(steps))}</div>'
            f'<div class="t">{t(f"nav.page.{key}")}</div><div class="d">{t(f"tour.text.{key}")}</div></div>')
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    with c1:
        st.button(t("tour.back"), on_click=_tour_move, args=(-1,), disabled=step == 0, width="stretch",
                  key="tour_back")
    with c2:
        last = step == len(steps) - 1
        st.button(t("tour.finish") if last else t("tour.next"), on_click=_tour_move, args=(1,), type="primary",
                  width="stretch", key="tour_next")
    with c3:
        st.button(t("tour.end"), on_click=lambda: st.session_state.update(tour_step=None), width="stretch",
                  key="tour_end")


# ---------------------------------------------------------------------------
# Presentation mode exit control, error boundary, footer
# ---------------------------------------------------------------------------
def render_present_exit() -> None:
    if st.session_state.get("present_mode"):
        st.button(t("sidebar.present_exit"), icon=":material/close_fullscreen:",
                  on_click=lambda: st.session_state.update(present_mode=False), key="present_exit")


def run_page(page) -> None:
    """Run the selected page inside a friendly error boundary."""
    try:
        page.run()
    except Exception as exc:  # st.rerun / st.stop are BaseException and pass through
        st.error(t("ui.error.title"), icon=":material/error:")
        st.caption(t("ui.error.hint"))
        with st.expander(t("ui.error.details")):
            st.exception(exc)


def render_footer() -> None:
    year = str(artifacts.get_metadata().get("created_utc", "2026"))[:4]
    st.html(f'<div class="hub-footer"><span>{t("footer.team")} · {t("app.name")} · {year}</span>'
            f'<span>{t("footer.disclaimer")}</span></div>')
