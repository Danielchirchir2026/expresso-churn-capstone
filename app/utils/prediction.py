"""Shared look and helpers for every page of the Expresso churn app.

Usage in a page:
    from app.utils.style import inject_css, page_header, kpi_row, insight, style_fig
    inject_css()
    page_header("📈", "Executive Dashboard", "Where churn comes from and where revenue sits.")
"""

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "ink": "#E8EEF7",
    "muted": "#8FA3BF",
    "line": "rgba(143, 163, 191, 0.22)",
    "churn": "#FF5D73",
    "stay": "#35D0BA",
    "understand": "#6EA8FE",
    "act": "#FFB454",
    "surface": "#16243A",
    "base": "#0F1A2B",
}

# Proposed names from the segment profiles. Edit freely.
SEGMENT_NAMES = {
    0: "Dormant",
    1: "Core loyalists",
    2: "Data-driven spenders",
    3: "Light users",
    4: "Fading users",
}

TONES = {
    "churn": COLORS["churn"],
    "stay": COLORS["stay"],
    "info": COLORS["understand"],
    "warn": COLORS["act"],
    "neutral": COLORS["ink"],
}


def segment_label(segment_id):
    """'Dormant (0)' style label for charts and tables."""
    return f"{SEGMENT_NAMES.get(int(segment_id), 'Segment')} ({int(segment_id)})"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@500;700;800&display=swap');
:root {
  --ink: #E8EEF7; --muted: #8FA3BF; --line: rgba(143,163,191,0.22);
  --churn: #FF5D73; --stay: #35D0BA; --understand: #6EA8FE; --act: #FFB454;
}
html, body, [class*="css"], .stMarkdown, .stMarkdown p { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
footer { visibility: hidden; }

.page-head { padding: 0.5rem 0 1.4rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.6rem; }
.page-head h1 {
  font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.3rem;
  letter-spacing: -0.02em; margin: 0 0 0.4rem 0; padding: 0; color: var(--ink);
}
.page-head p { color: var(--muted); font-size: 1.05rem; line-height: 1.6; margin: 0; max-width: 46rem; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin: 0.2rem 0 1.8rem 0;
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.kpi { padding: 1.1rem 1.3rem; border-left: 1px solid var(--line); }
.kpi:first-child { border-left: none; padding-left: 0; }
.kpi .label { color: var(--muted); font-size: 0.92rem; }
.kpi .value { font-family: 'Sora', sans-serif; font-size: 1.85rem; font-weight: 700; line-height: 1.25; }
.kpi .note { color: var(--muted); font-size: 0.85rem; }

.insight {
  padding: 0.85rem 1.1rem; margin: 0.5rem 0 0.8rem 0;
  border-left: 3px solid var(--tone, var(--understand));
  background: rgba(143,163,191,0.07); border-radius: 0 10px 10px 0;
  color: var(--ink); line-height: 1.55;
}
.section-title {
  font-family: 'Sora', sans-serif; font-size: 1.25rem; font-weight: 700;
  color: var(--ink); margin: 2rem 0 0.2rem 0;
}
.section-sub { color: var(--muted); font-size: 0.95rem; margin-bottom: 0.8rem; }
.verdict {
  padding: 1.2rem 1.4rem; border: 1px solid var(--line); border-radius: 12px;
  border-left: 4px solid var(--tone, var(--understand)); background: rgba(143,163,191,0.07);
}
.verdict b { font-family: 'Sora', sans-serif; font-size: 2.2rem; color: var(--tone, var(--ink)); }
.verdict span { display: block; color: var(--muted); margin-top: 0.2rem; }
</style>
"""


def inject_css():
    """Call once at the top of every page, right after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Building blocks (HTML has no indentation or blank lines on purpose)
# ---------------------------------------------------------------------------
def page_header(icon, title, subtitle):
    st.markdown(
        f'<div class="page-head"><h1>{icon} {title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):
    html = f'<div class="section-title">{title}</div>'
    if subtitle:
        html += f'<div class="section-sub">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, note, tone). Tone: churn, stay, info, warn, neutral."""
    cells = []
    for label, value, note, tone in items:
        colour = TONES.get(tone, COLORS["ink"])
        cells.append(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value" style="color:{colour}">{value}</div>'
            f'<div class="note">{note}</div></div>'
        )
    st.markdown(f'<div class="kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def insight(text, tone="info"):
    colour = TONES.get(tone, COLORS["understand"])
    st.markdown(
        f'<div class="insight" style="--tone:{colour}">{text}</div>',
        unsafe_allow_html=True,
    )


def verdict(value, caption, tone="info"):
    colour = TONES.get(tone, COLORS["understand"])
    st.markdown(
        f'<div class="verdict" style="--tone:{colour}"><b>{value}</b><span>{caption}</span></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=380):
    """Apply the shared Plotly look: transparent background, muted grid, DM Sans."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=COLORS["ink"], size=14),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=COLORS["surface"], font_size=13),
    )
    fig.update_xaxes(gridcolor="rgba(143,163,191,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(143,163,191,0.12)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Model insights: churn drivers read straight from the deployed model
# ---------------------------------------------------------------------------
def _theme_of(feature):
    f = feature.upper()
    if f.startswith("REGION"):
        return "Region"
    if "MISSING" in f:
        return "Data gaps"
    if any(k in f for k in ("MONTANT", "FREQUENCE_RECH", "PER_RECHARGE", "TOP_PACK")):
        return "Recharge behaviour"
    if any(k in f for k in ("REVENUE", "ARPU")):
        return "Revenue"
    if "DATA" in f:
        return "Data usage"
    if any(k in f for k in ("ON_NET", "ORANGE", "TIGO", "ZONE")):
        return "Network mix"
    if any(k in f for k in ("FREQUENCE", "REGULARITY", "TENURE")):
        return "Activity and loyalty"
    return "Other"


@st.cache_resource(show_spinner=False)
def _bundle():
    try:
        from app.utils.prediction import load_bundle
    except ImportError:  # fall back if the repo uses a different import root
        from utils.prediction import load_bundle
    return load_bundle()


@st.cache_data(show_spinner=False)
def driver_importance():
    """Return a DataFrame of features, gain importance (as % of total) and theme."""
    bundle = _bundle()
    model = bundle["model"]
    names = bundle["preprocessing_config"]["feature_columns"]
    try:
        gains = model.booster_.feature_importance(importance_type="gain")
    except AttributeError:
        gains = model.feature_importances_
    df = pd.DataFrame({"feature": names, "gain": gains})
    df["importance_pct"] = df["gain"] / df["gain"].sum() * 100
    df["theme"] = df["feature"].map(_theme_of)
    return df.sort_values("importance_pct", ascending=False).reset_index(drop=True)
