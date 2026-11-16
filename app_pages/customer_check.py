"""Customer Check: score one customer, explain it, suggest a next action."""

import html
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import navigation
from app.core import artifacts, explain, fmt, portfolio, presets, scoring, state
from app.core.features import friendly
from app.core.segments import SEGMENT_IDS, segment_name
from app.i18n import t
from app.ui import charts, layout
from app.ui.components import insight_callout, kpi_row, page_header, prob_gauge, section_header

bundle = artifacts.get_bundle()
config = bundle["preprocessing_config"]
profiles = artifacts.get_segment_profiles()
pf = portfolio.compute(profiles, artifacts.get_metadata())
eda = artifacts.get_optional("eda_summary")
NA = "__not_recorded__"
NUMERIC = presets.NUMERIC
MISSABLE = set(config["columns_with_missing_values"])
REGIONS = [r for r in config["region_values"] if r != "MISSING"]
PACKS = [k for k in config["top_pack_frequency"] if k not in ("MISSING", "Other")]

page_header(t("cc.title"), t("cc.sub"), t("cc.eyebrow"))


# ---------------------------------------------------------------------------
# Form state: canonical values live in st.session_state["cc_values"]
# ---------------------------------------------------------------------------
def _default_values() -> dict:
    vals, _ = presets.preset(3, profiles, eda)
    base = {"user_id": "CUSTOMER-001", "REGION": "DAKAR", "TENURE": config["tenure_order"][-1], "MRG": "NO",
            "TOP_PACK": np.nan}
    base.update(vals)
    return base


def _push_to_widgets(values: dict) -> None:
    for c in NUMERIC:
        if c == "REGULARITY":
            continue
        v = values.get(c)
        missing = v is None or (isinstance(v, float) and math.isnan(v))
        st.session_state[f"cc_{c}"] = 0.0 if missing else float(v)
        if c in MISSABLE:
            st.session_state[f"cc_{c}_na"] = bool(missing)
    reg = values.get("REGULARITY")
    reg = 0 if reg is None or (isinstance(reg, float) and math.isnan(reg)) else reg
    st.session_state["cc_REGULARITY"] = int(min(max(round(float(reg)), 0), 90))
    region = values.get("REGION")
    st.session_state["cc_REGION"] = region if isinstance(region, str) and region in REGIONS else NA
    st.session_state["cc_TENURE"] = values.get("TENURE") or config["tenure_order"][-1]
    pack = values.get("TOP_PACK")
    st.session_state["cc_TOP_PACK"] = pack if isinstance(pack, str) and pack in PACKS + ["Other"] else NA
    st.session_state["cc_user_id"] = values.get("user_id", "CUSTOMER-001")


def apply_preset(seg_id: int) -> None:
    vals, sourced = presets.preset(seg_id, profiles, eda)
    current = dict(st.session_state.get("cc_values") or _default_values())
    keep = {k: current.get(k) for k in ("user_id", "REGION", "TENURE", "MRG")}
    new = {"TOP_PACK": np.nan, **keep, **vals}
    st.session_state["cc_values"] = new
    st.session_state["cc_preset"] = (seg_id, sorted(sourced))
    _push_to_widgets(new)
    st.session_state["cc_has_run"] = True


if "cc_values" not in st.session_state:
    st.session_state["cc_values"] = _default_values()
    st.session_state["cc_preset"] = (3, sorted(presets.preset(3, profiles, eda)[1]))
    st.session_state["cc_has_run"] = True  # open with a scored example
if any(f"cc_{c}" not in st.session_state for c in NUMERIC):
    _push_to_widgets(st.session_state["cc_values"])

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
section_header(t("cc.presets.title"), t("cc.presets.sub"))
pcols = st.columns(len(SEGMENT_IDS))
for col, sid in zip(pcols, SEGMENT_IDS):
    with col:
        st.button(t("cc.presets.button", name=segment_name(sid)), key=f"cc_preset_{sid}", width="stretch",
                  on_click=apply_preset, args=(sid,), help=t("cc.presets.help"))
seg_used, sourced = st.session_state.get("cc_preset", (None, []))
if seg_used is not None:
    filled = ", ".join(friendly(c) for c in sourced) or "–"
    note = t("cc.presets.note_eda") if eda and eda.get("segment_medians") else t("cc.presets.note_profile")
    st.caption(t("cc.presets.loaded", name=segment_name(seg_used), fields=filled) + " " + note)

# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
HELP = {c: t(f"feat.{c}.desc") for c in config["raw_source_columns"]}


def number(col: str, step: float = 1.0) -> None:
    st.number_input(f"{friendly(col)} ({col})", min_value=0.0, step=step, key=f"cc_{col}", help=HELP.get(col))
    if col in MISSABLE:
        st.checkbox(t("cc.form.not_recorded"), key=f"cc_{col}_na", help=t("cc.form.not_recorded_help"))


with st.form("cc_form", border=True):
    tabs = st.tabs([t("cc.tab.profile"), t("cc.tab.recharge"), t("cc.tab.usage"), t("cc.tab.activity")])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input(t("cc.form.customer_id"), key="cc_user_id", help=t("cc.form.customer_id_help"))
        with c2:
            st.selectbox(f"{friendly('REGION')} (REGION)", [NA] + REGIONS, key="cc_REGION", help=HELP["REGION"],
                         format_func=lambda v: t("cc.form.not_recorded_opt") if v == NA else v)
        with c3:
            st.selectbox(f"{friendly('TENURE')} (TENURE)", config["tenure_order"], key="cc_TENURE",
                         help=HELP["TENURE"])
        st.caption(t("cc.form.mrg_note"))
    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            number("MONTANT", 100.0)
            number("ARPU_SEGMENT", 50.0)
        with c2:
            number("FREQUENCE_RECH")
            number("FREQUENCE")
        with c3:
            number("REVENUE", 100.0)
    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        with c1:
            number("DATA_VOLUME", 10.0)
            number("ZONE1")
        with c2:
            number("ON_NET")
            number("ZONE2")
        with c3:
            number("ORANGE")
            number("TIGO")
    with tabs[3]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.slider(f"{friendly('REGULARITY')} (REGULARITY)", 0, 90, key="cc_REGULARITY", help=HELP["REGULARITY"])
        with c2:
            st.selectbox(f"{friendly('TOP_PACK')} (TOP_PACK)", [NA] + PACKS + ["Other"], key="cc_TOP_PACK",
                         help=HELP["TOP_PACK"],
                         format_func=lambda v: t("cc.form.not_recorded_opt") if v == NA else (
                             t("cc.form.other_pack") if v == "Other" else v))
        with c3:
            number("FREQ_TOP_PACK")
    submitted = st.form_submit_button(t("cc.form.submit"), type="primary", width="stretch",
                                      icon=":material/query_stats:")

if submitted:
    vals = {"user_id": (st.session_state["cc_user_id"] or "CUSTOMER-001").strip(), "MRG": "NO",
            "TENURE": st.session_state["cc_TENURE"]}
    vals["REGION"] = np.nan if st.session_state["cc_REGION"] == NA else st.session_state["cc_REGION"]
    vals["TOP_PACK"] = np.nan if st.session_state["cc_TOP_PACK"] == NA else st.session_state["cc_TOP_PACK"]
    for c in NUMERIC:
        if c in MISSABLE and st.session_state.get(f"cc_{c}_na"):
            vals[c] = np.nan
        else:
            vals[c] = float(st.session_state[f"cc_{c}"])
    st.session_state["cc_values"] = vals
    st.session_state["cc_has_run"] = True

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
def render_result() -> None:
    raw = dict(st.session_state["cc_values"])
    state.set_last_customer(raw)
    res = scoring.score_one(raw, bundle)
    prob = res["probability"]
    seg = res["segment"]
    seg_rate = float(profiles.set_index("segment").loc[seg, "churn_rate"])

    section_header(t("cc.result.title", id=raw.get("user_id", "")), t("cc.result.sub"))
    left, right = st.columns([1, 1.35])
    with left:
        prob_gauge(prob, reference=pf.churn_rate / 100, color=charts.risk_up() if prob * 100 > pf.churn_rate else None)
        gap = prob * 100 - pf.churn_rate
        kpi_row([
            dict(label=t("cc.result.vs_portfolio"), value=fmt.pp(gap, 1),
                 delta=t("cc.result.portfolio_is", rate=fmt.pct(pf.churn_rate, 2)),
                 tone="bad" if gap > 0 else "good"),
            dict(label=t("cc.result.multiple"), value=f"{fmt.num(prob * 100 / pf.churn_rate, 2)}×",
                 help=t("cc.result.multiple_help")),
            dict(label=t("cc.result.segment"), value=segment_name(seg),
                 foot=t("cc.result.segment_rate", rate=fmt.pct(seg_rate, 2)), help=t("cc.result.segment_help")),
        ], cols=1)
        st.caption(t("cc.result.no_bands"))

    with right:
        show_raw = st.toggle(t("cc.why.raw_toggle"), key="cc_raw_toggle", help=t("cc.why.raw_help"))
        series = res["raw"] if show_raw else res["grouped"]
        top = series.head(8)
        rest = float(series.iloc[8:].sum())
        bias = res["bias"]
        labels = [t("cc.why.start")] + [(f if show_raw else friendly(f)) for f in top.index] + [t("cc.why.others"),
                                                                                                  t("cc.why.end")]
        measures = ["absolute"] + ["relative"] * (len(top) + 1) + ["total"]
        # Anchored at the model baseline: the first bar has zero length and the
        # total bar spans the net move from the average customer to this one.
        values = [0.0] + list(top.values) + [rest, 0.0]
        cum = np.cumsum([bias] + list(top.values) + [rest])
        hover_contrib = [0.0] + list(top.values) + [rest, float(cum[-1] - bias)]
        logit = lambda q: math.log(q / (1 - q))  # noqa: E731
        ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
        lo, hi = float(min(cum.min(), bias)) - 0.6, float(max(cum.max(), bias)) + 0.6
        tv = [logit(q) for q in ticks if lo <= logit(q) <= hi]
        tt = [fmt.pct(q * 100, 1 if q < 0.02 or q > 0.98 else 0) for q in ticks if lo <= logit(q) <= hi]
        after = list(cum) + [cum[-1]]
        hover_p = [1 / (1 + math.exp(-x)) * 100 for x in after]
        wf = go.Figure(go.Waterfall(
            orientation="h", y=labels, x=values, measure=measures, base=bias,
            increasing=dict(marker=dict(color=charts.risk_up())), decreasing=dict(marker=dict(color=charts.risk_down())),
            totals=dict(marker=dict(color=charts.accent())), connector=dict(line=dict(color=charts.ink("axis"), width=1)),
            customdata=np.array([hover_contrib, hover_p]).T,
            hovertemplate="<b>%{y}</b><br>" + t("cc.why.hover_contrib") + ": %{customdata[0]:+.2f}<br>"
                          + t("cc.why.hover_after") + ": %{customdata[1]:.1f}%<extra></extra>",
        ))
        charts.style(wf, height=440, legend=False, xgrid=True, ygrid=False, title=t("cc.why.title"))
        wf.update_yaxes(autorange="reversed")
        wf.add_vline(x=bias, line=dict(color=charts.ink("muted"), width=1))
        wf.update_xaxes(tickvals=tv, ticktext=tt, range=[lo, hi], title=t("cc.why.axis"))
        first = top.index[0] if len(top) else None
        charts.show(
            wf,
            t("cc.why.so_what", name=(first if show_raw else friendly(first)) if first else "–",
              dir=t("cc.why.dir_up") if (first and top.iloc[0] > 0) else t("cc.why.dir_down")),
            data=pd.DataFrame({t("cc.why.col_factor"): labels[1:-2], t("cc.why.hover_contrib"): np.round(top.values, 3)}),
            key="cc_waterfall",
        )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        section_header(t("cc.reasons.title"), t("cc.reasons.sub"))
        items = explain.reason_sentences(res["grouped"], raw, res["features"], config, n=5)
        rows = "".join(
            f'<div class="hub-reason"><span class="dir {d}">{"▲" if d == "up" else "▼"}</span>'
            f'<span>{html.escape(s)}</span></div>' for d, s in items)
        st.html(f'<div class="hub-card">{rows}</div>')
    with c2:
        section_header(t("cc.nba.title"), t("cc.nba.sub"))
        ctx = explain.build_context(prob, res["grouped"], raw, res["features"], pf.churn_rate / 100, profiles)
        for key in explain.next_best_actions(ctx):
            insight_callout(t(f"nba.{key}.body"), "good" if key == "low_risk" else "info", tag=t(f"nba.{key}.title"))
        st.caption(t("cc.nba.caption"))

    if navigation.available("what_if"):
        st.button(t("cc.to_whatif"), icon=":material/tune:", type="primary", on_click=layout.request_goto,
                  args=("what_if",), key="cc_to_whatif")


if st.session_state.get("cc_has_run"):
    render_result()
else:
    st.info(t("cc.hint_run"), icon=":material/touch_app:")
