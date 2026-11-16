"""Home: hero, headline KPIs, persona routes, team."""

import html

import streamlit as st

from app import navigation
from app.core import artifacts, fmt, portfolio
from app.i18n import t
from app.ui import charts, layout
from app.ui.components import churn_ring, kpi_row, raw_html, section_header

profiles = artifacts.get_segment_profiles()
meta = artifacts.get_metadata()
pf = portfolio.compute(profiles, meta)

TEAM = ["Daniel Chirchir", "Daniel Tabut", "Livingstone Talel", "Sandra Bosibori", "Patrick Palex",
        "Immanuel Ochieng"]

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
chips = "".join(
    f'<span class="hub-chip">{html.escape(text)}</span>'
    for text in (
        t("home.chip.model", model=meta["model_name"]),
        t("home.chip.features", n=meta["feature_count"]),
        t("home.chip.segments", n=len(profiles)),
        t("home.chip.rows", n=fmt.compact(meta["training_rows"])),
    )
)
ring = churn_ring(
    pf.churn_rate / 100,
    fmt.pct(pf.churn_rate, 2),
    t("home.ring_label"),
    charts.risk_up(),
    charts.risk_down(),
)
raw_html(
    f'<div class="hub-hero"><div>'
    f'<div class="hub-eyebrow">{html.escape(t("home.eyebrow"))}</div>'
    f'<h1>{html.escape(t("home.title"))}</h1>'
    f'<p class="lede">{html.escape(t("home.lede"))}</p>'
    f'<div class="hub-chips">{chips}</div></div>'
    f'<div>{ring}<div style="display:flex;justify-content:center;gap:1rem;margin-top:.7rem;font-size:.85rem;opacity:.85">'
    f'<span><span style="display:inline-block;width:.6rem;height:.6rem;border-radius:50%;background:{charts.risk_up()};'
    f'margin-right:.35rem"></span>{html.escape(t("home.ring_churned"))} {fmt.num(pf.est_churners)}</span>'
    f'<span><span style="display:inline-block;width:.6rem;height:.6rem;border-radius:50%;background:{charts.risk_down()};'
    f'margin-right:.35rem"></span>{html.escape(t("home.ring_stayed"))} {fmt.pct(100 - pf.churn_rate, 2)}</span>'
    f'</div></div></div>'
)

kpi_row([
    dict(label=t("kpi.total_customers"), value=fmt.num(pf.total_customers), help=t("kpi.total_customers_help")),
    dict(label=t("kpi.churn_rate"), value=fmt.pct(pf.churn_rate, 2), help=t("kpi.churn_rate_help")),
    dict(label=t("kpi.est_churners"), value=fmt.num(pf.est_churners), help=t("kpi.est_churners_help")),
    dict(label=t("kpi.roc_auc"), value=fmt.num(pf.roc_auc, 4), help=t("kpi.roc_auc_help")),
])

# ---------------------------------------------------------------------------
# Start here: one route per persona
# ---------------------------------------------------------------------------
section_header(t("home.start.title"), t("home.start.sub"))

PERSONAS = [
    ("executive", "executive", ["executive", "revenue_risk", "problem"]),
    ("retention", "retention" if navigation.available("retention") else "batch",
     ["retention", "batch", "campaigns"]),
    ("analyst", "model", ["model", "evaluation", "data_explorer"]),
]

cols = st.columns(3)
for col, (role, target, links) in zip(cols, PERSONAS):
    with col:
        with st.container(border=True, height="stretch"):
            st.html(f'<div class="hub-persona"><div class="who">{html.escape(t(f"home.persona.{role}.who"))}</div>'
                    f'<h4>{html.escape(t(f"home.persona.{role}.title"))}</h4>'
                    f'<p>{html.escape(t(f"home.persona.{role}.desc"))}</p></div>')
            st.button(t(f"home.persona.{role}.cta"), key=f"persona_{role}", type="primary", width="stretch",
                      icon=":material/arrow_forward:", on_click=layout.request_goto, args=(target, role))
            for key in links:
                if key != target:
                    navigation.link(key)

c1, c2 = st.columns([1, 3])
with c1:
    if st.button(t("home.tour_cta"), icon=":material/explore:", width="stretch", key="home_tour"):
        layout.start_tour()
with c2:
    st.caption(t("home.tour_hint"))

# ---------------------------------------------------------------------------
# What's inside
# ---------------------------------------------------------------------------
section_header(t("home.inside.title"), t("home.inside.sub"))
sections = ["overview", "predict", "act", "trust", "assistant", "reports"]
grid = st.columns(3)
for i, section in enumerate(sections):
    keys = [k for k, (_, _, s) in navigation.PAGES.items() if s == section and navigation.available(k)]
    if not keys:
        continue
    with grid[i % 3]:
        with st.container(border=True):
            st.markdown(f"**{t(f'nav.section.{section}')}**")
            for key in keys:
                navigation.link(key, help=t(f"pages.{key}"))

# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------
section_header(t("home.team.title"), t("home.team.sub"))
st.html('<div class="hub-chips">' + "".join(f'<span class="hub-chip">{html.escape(n)}</span>' for n in TEAM)
        + "</div>")
st.caption(t("home.provenance"))
