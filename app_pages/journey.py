"""Customer Journey: follow one customer from lookup to action on a single page.

1 Find the customer -> 2 How likely they are to leave -> 3 Why -> 4 What would change it -> 5 What to do.
Steps 2 to 5 always describe the customer chosen in step 1. That customer's raw values live in
st.session_state["cc_values"]; the hand-entry form widgets (cc_*) only mirror them.
"""

import html
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import navigation
from app.core import artifacts, batchops, explain, fmt, portfolio, presets, scoring, state
from app.core.features import friendly
from app.core.segments import SEGMENT_IDS, segment_name
from app.i18n import t
from app.ui import charts
from app.ui.components import (insight_callout, journey_strip, kpi_row, page_header, prob_gauge, step_header,
                               synthetic_banner)

bundle = artifacts.get_bundle()
config = bundle["preprocessing_config"]
ID = config["id_column"]
profiles = artifacts.get_segment_profiles()
pf = portfolio.compute(profiles, artifacts.get_metadata())
eda = artifacts.get_optional("eda_summary")
NA = "__not_recorded__"
NUMERIC = presets.NUMERIC
MISSABLE = set(config["columns_with_missing_values"])
REGIONS = [r for r in config["region_values"] if r != "MISSING"]
PACKS = [k for k in config["top_pack_frequency"] if k not in ("MISSING", "Other")]
TENURES = list(config["tenure_order"])
DEFAULT_ID = "CUSTOMER-001"
DEFAULT_SEGMENT = 0  # the typical Silent / Inactive customer: a journey with real reasons and actions
PROFILE_FIELDS = ["REGION", "TENURE", "REGULARITY", "REVENUE", "FREQUENCE_RECH", "DATA_VOLUME"]

MAIN = ["REGULARITY", "FREQUENCE_RECH", "MONTANT", "REVENUE", "DATA_VOLUME", "ON_NET"]
MORE = ["ARPU_SEGMENT", "FREQUENCE", "ORANGE", "TIGO", "FREQ_TOP_PACK"]
LEVERS = MAIN + MORE
SCALE_OF = {"MONTANT": "REVENUE", "ARPU_SEGMENT": "REVENUE", "FREQUENCE": "FREQUENCE_RECH",
            "FREQ_TOP_PACK": "FREQUENCE_RECH"}


def is_na(v) -> bool:
    try:
        return v is None or bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


page_header(t("jr.title"), t("jr.sub"), t("jr.eyebrow"))
strip = st.empty()  # the at-a-glance strip; filled at the end, once every step has its numbers


# ===========================================================================
# 1 · Who is the customer?
# ===========================================================================
def _default_values() -> dict:
    vals, _ = presets.preset(DEFAULT_SEGMENT, profiles, eda)
    return {ID: DEFAULT_ID, "REGION": "DAKAR", "TENURE": TENURES[-1], "MRG": "NO", "TOP_PACK": np.nan, **vals}


def _push_to_widgets(values: dict) -> None:
    """Mirror the customer's values into the hand-entry form."""
    ss = st.session_state
    for c in NUMERIC:
        v = values.get(c)
        if c == "REGULARITY":
            ss["cc_REGULARITY"] = 0 if is_na(v) else int(min(max(round(float(v)), 0), 90))
            continue
        ss[f"cc_{c}"] = 0.0 if is_na(v) else max(float(v), 0.0)
        if c in MISSABLE:
            ss[f"cc_{c}_na"] = is_na(v)
    region, tenure, pack = values.get("REGION"), values.get("TENURE"), values.get("TOP_PACK")
    ss["cc_REGION"] = region if region in REGIONS else NA
    ss["cc_TENURE"] = tenure if tenure in TENURES else TENURES[-1]
    ss["cc_TOP_PACK"] = NA if is_na(pack) else (pack if pack in PACKS else "Other")
    ss["cc_user_id"] = str(values.get(ID) or DEFAULT_ID)


def _load(values: dict, origin: dict) -> None:
    st.session_state["cc_values"] = values
    st.session_state["jr_origin"] = origin
    st.session_state.pop("wi_signature", None)  # an explicit load always restarts the what-if levers
    _push_to_widgets(values)


def load_preset(seg_id: int) -> None:
    vals, sourced = presets.preset(seg_id, profiles, eda)
    from_batch = st.session_state.get("jr_origin", {}).get("kind") == "batch"
    base = _default_values() if from_batch else (st.session_state.get("cc_values") or _default_values())
    keep = {k: base.get(k) for k in (ID, "REGION", "TENURE", "MRG")}
    _load({"TOP_PACK": np.nan, **keep, **vals}, {"kind": "preset", "segment": seg_id, "sourced": sorted(sourced)})


def load_from_batch(pick_key: str) -> None:
    batch = state.get_batch()
    if not batch:
        return
    df = batch["df"]
    target = (st.session_state.get("jr_batch_search") or "").strip() or st.session_state.get(pick_key)
    hit = df[df[ID].astype(str) == str(target)]
    if hit.empty:
        st.session_state["jr_batch_miss"] = str(target)
        return
    st.session_state.pop("jr_batch_miss", None)
    st.session_state["jr_batch_search"] = ""
    row = hit.iloc[0]
    vals = {c: (np.nan if is_na(row.get(c)) else row.get(c)) for c in config["raw_source_columns"]}
    vals[ID] = str(row[ID])
    _load(vals, {"kind": "batch", "name": batch["name"]})


if "cc_values" not in st.session_state or "jr_origin" not in st.session_state:
    _load(_default_values(), {"kind": "preset", "segment": DEFAULT_SEGMENT,
                              "sourced": sorted(presets.preset(DEFAULT_SEGMENT, profiles, eda)[1])})
elif any(f"cc_{c}" not in st.session_state for c in NUMERIC):
    _push_to_widgets(st.session_state["cc_values"])

step_header(1, t("jr.s1.title"), t("jr.s1.sub"), "jr-find")
SOURCES = {"preset": t("jr.s1.src_preset"), "batch": t("jr.s1.src_batch"), "manual": t("jr.s1.src_manual")}
source = st.segmented_control(t("jr.s1.how"), list(SOURCES), format_func=SOURCES.__getitem__, default="preset",
                              required=True, key="jr_source")

if source == "batch":
    batch = state.get_batch()
    if batch:
        if batch["synthetic"]:
            synthetic_banner()
        df_b = batch["df"]
        riskiest = df_b.nlargest(50, "churn_probability")
        picks = {str(i): f"{i} · {fmt.pct(p * 100, 1)}" for i, p in zip(riskiest[ID], riskiest["churn_probability"])}
        pick_key = f"jr_batch_pick_{batch['id']}"  # a new batch gets a fresh list
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(t("jr.s1.batch_pick"), list(picks), format_func=picks.__getitem__, key=pick_key)
        with c2:
            st.text_input(t("batch.c360.search"), key="jr_batch_search", placeholder=t("batch.c360.placeholder"))
        st.button(t("jr.s1.batch_go"), type="primary", icon=":material/person_search:", on_click=load_from_batch,
                  args=(pick_key,), key="jr_batch_go")
        if st.session_state.get("jr_batch_miss"):
            st.warning(t("batch.c360.not_found", id=st.session_state["jr_batch_miss"]))
        st.caption(t("jr.s1.batch_note", name=batch["name"], n=fmt.num(batch["rows"])))
    else:
        st.info(t("jr.s1.batch_none"), icon=":material/info:")
        c1, c2 = st.columns([1, 2])  # wide enough for the full button label
        with c1:
            navigation.link("batch", label=t("batchui.empty.go"), icon=":material/upload_file:")
        with c2:
            if st.button(t("batchui.empty.demo"), key="jr_quick_demo", type="primary", icon=":material/bolt:"):
                bar = st.progress(0.0, text=t("batch.score.progress"))
                batchops.quick_demo(5000, progress=lambda f: bar.progress(f, text=t("batch.score.progress")))
                st.rerun()

elif source == "manual":
    HELP = {c: t(f"feat.{c}.desc") for c in config["raw_source_columns"]}

    def number(col: str, step: float = 1.0) -> None:
        st.number_input(f"{friendly(col)} ({col})", min_value=0.0, step=step, key=f"cc_{col}", help=HELP.get(col))
        if col in MISSABLE:
            st.checkbox(t("cc.form.not_recorded"), key=f"cc_{col}_na", help=t("cc.form.not_recorded_help"))

    st.caption(t("jr.s1.manual_note"))
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
                st.selectbox(f"{friendly('TENURE')} (TENURE)", TENURES, key="cc_TENURE", help=HELP["TENURE"])
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
                st.slider(f"{friendly('REGULARITY')} (REGULARITY)", 0, 90, key="cc_REGULARITY",
                          help=HELP["REGULARITY"])
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
        ss = st.session_state
        vals = {ID: (ss["cc_user_id"] or DEFAULT_ID).strip(), "MRG": "NO", "TENURE": ss["cc_TENURE"]}
        vals["REGION"] = np.nan if ss["cc_REGION"] == NA else ss["cc_REGION"]
        vals["TOP_PACK"] = np.nan if ss["cc_TOP_PACK"] == NA else ss["cc_TOP_PACK"]
        for c in NUMERIC:
            vals[c] = np.nan if (c in MISSABLE and ss.get(f"cc_{c}_na")) else float(ss[f"cc_{c}"])
        st.session_state["cc_values"] = vals
        st.session_state["jr_origin"] = {"kind": "manual"}
        st.session_state.pop("wi_signature", None)

else:  # typical customer of a segment
    origin = st.session_state["jr_origin"]
    pcols = st.columns(len(SEGMENT_IDS))
    for col, sid in zip(pcols, SEGMENT_IDS):
        with col:
            active = origin.get("kind") == "preset" and origin.get("segment") == sid
            st.button(t("cc.presets.button", name=segment_name(sid)), key=f"jr_preset_{sid}", width="stretch",
                      type="primary" if active else "secondary", on_click=load_preset, args=(sid,),
                      help=t("cc.presets.help"))
    st.caption(t("cc.presets.sub"))

raw = dict(st.session_state["cc_values"])
cust_id = str(raw.get(ID) or DEFAULT_ID)
origin = st.session_state["jr_origin"]
if origin["kind"] == "preset":
    how = t("jr.origin.preset", name=segment_name(origin["segment"]))
    how_short = t("jr.strip.find_preset", name=segment_name(origin["segment"]))
elif origin["kind"] == "batch":
    how, how_short = t("jr.origin.batch", name=origin["name"]), t("jr.strip.find_batch")
else:
    how = how_short = t("jr.origin.manual")


def shown(col: str) -> str:
    v = raw.get(col)
    if is_na(v):
        return t("cc.form.not_recorded")
    return fmt.num(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


chips = "".join(f"<span>{html.escape(friendly(c))}: <b>{html.escape(shown(c))}</b></span>" for c in PROFILE_FIELDS)
st.html(f'<div class="hub-card hub-who"><div class="hub-kpi-label">{html.escape(t("jr.s1.following"))}</div>'
        f'<div class="id">{html.escape(cust_id)}</div><div class="how">{html.escape(how)}</div>'
        f'<div class="hub-profile">{chips}</div></div>')
if origin["kind"] == "preset":
    filled = ", ".join(friendly(c) for c in origin.get("sourced", [])) or "–"
    note = t("cc.presets.note_eda") if eda and eda.get("segment_medians") else t("cc.presets.note_profile")
    st.caption(t("cc.presets.loaded", name=segment_name(origin["segment"]), fields=filled) + " " + note)

# ===========================================================================
# 2 · How likely is this customer to leave?
# ===========================================================================
state.set_last_customer(raw)  # the Copilot's what-if tool starts from this customer
res = scoring.score_one(raw, bundle)
prob = res["probability"]
seg = res["segment"]
seg_rate = float(profiles.set_index("segment").loc[seg, "churn_rate"])
gap = prob * 100 - pf.churn_rate
mult = prob * 100 / pf.churn_rate

step_header(2, t("jr.s2.title"), t("jr.s2.sub"), "jr-risk")
left, right = st.columns([1, 1.4])
with left:
    prob_gauge(prob, reference=pf.churn_rate / 100, color=charts.risk_up() if gap > 0 else None)
with right:
    above = gap > 0
    insight_callout(t("jr.s2.verdict_up" if above else "jr.s2.verdict_down", id=cust_id, p=fmt.pct(prob * 100, 1),
                      mult=fmt.num(mult, 2), rate=fmt.pct(pf.churn_rate, 2)),
                    "risk" if above else "good", tag=t("jr.s2.tag_up" if above else "jr.s2.tag_down"))
    kpi_row([
        dict(label=t("cc.result.vs_portfolio"), value=fmt.pp(gap, 1),
             delta=t("cc.result.portfolio_is", rate=fmt.pct(pf.churn_rate, 2)), tone="bad" if above else "good"),
        dict(label=t("cc.result.multiple"), value=f"{fmt.num(mult, 2)}×", help=t("cc.result.multiple_help")),
        dict(label=t("cc.result.segment"), value=segment_name(seg),
             foot=t("cc.result.segment_rate", rate=fmt.pct(seg_rate, 2)), help=t("cc.result.segment_help")),
    ], cols=3)
    st.caption(t("cc.result.no_bands"))

# ===========================================================================
# 3 · Why?
# ===========================================================================
step_header(3, t("jr.s3.title"), t("jr.s3.sub"), "jr-why")
c1, c2 = st.columns([1.35, 1])
with c1:
    show_raw = st.toggle(t("cc.why.raw_toggle"), key="jr_raw_toggle", help=t("cc.why.raw_help"))
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
    hover_p = [1 / (1 + math.exp(-x)) * 100 for x in list(cum) + [cum[-1]]]
    wf = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measures, base=bias,
        increasing=dict(marker=dict(color=charts.risk_up())), decreasing=dict(marker=dict(color=charts.risk_down())),
        # Totals in graphite: the accent is also blue, like the "lowers risk" bars.
        totals=dict(marker=dict(color=charts.ink("secondary"))),
        connector=dict(line=dict(color=charts.ink("axis"), width=1)),
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
        key="jr_waterfall",
    )
with c2:
    st.markdown(f"**{t('cc.reasons.title')}**")
    st.caption(t("cc.reasons.sub"))
    reasons = explain.reason_sentences(res["grouped"], raw, res["features"], config, n=5)
    rows = "".join(f'<div class="hub-reason"><span class="dir {d}">{"▲" if d == "up" else "▼"}</span>'
                   f'<span>{html.escape(s)}</span></div>' for d, s in reasons)
    st.html(f'<div class="hub-card">{rows}</div>')

# ===========================================================================
# 4 · What would change it?
# ===========================================================================
step_header(4, t("jr.s4.title"), t("jr.s4.sub"), "jr-whatif")
baseline = dict(raw)


def reset_levers() -> None:
    for c in LEVERS:
        st.session_state.pop(f"wi_{c}", None)
        st.session_state.pop(f"wi_{c}_na", None)


signature = repr(sorted((k, str(v)) for k, v in baseline.items()))
if st.session_state.get("wi_signature") != signature:  # new customer: levers start from their values
    reset_levers()
    st.session_state["wi_signature"] = signature


def lever_max(col: str) -> float:
    """Slider range: training distribution when exported, else scaled with the customer."""
    if col == "REGULARITY":
        return 90.0
    base = baseline.get(col)
    base = 0.0 if is_na(base) else float(base)
    stats = ((eda or {}).get("numeric") or {}).get(col, {}).get("all")
    if stats:
        iqr = stats["q3"] - stats["q1"]
        upper = min(stats["max"], stats["q3"] + 3 * iqr)
        return float(max(upper, base * 1.5, 1.0))
    ref_col = SCALE_OF.get(col, col)
    prof_col = presets.FROM_PROFILE.get(ref_col)
    ref = float(profiles[prof_col].max()) if prof_col else 0.0
    return float(max(3 * base, 3 * ref, 500.0 if col in ("ON_NET", "ORANGE", "TIGO") else 50.0))


ranges = {c: lever_max(c) for c in LEVERS}
step_of = lambda m: 1.0 if m <= 200 else (10.0 if m <= 5000 else 100.0)  # noqa: E731
scenario = dict(baseline)


def lever(col: str) -> None:
    b = baseline.get(col)
    mx = ranges[col]
    if col == "REGULARITY":
        default = int(0 if is_na(b) else min(max(round(float(b)), 0), 90))
        v = st.slider(f"{friendly(col)}", 0, 90, value=default, key=f"wi_{col}", help=t(f"feat.{col}.desc"))
        scenario[col] = float(v)
        return
    keep_na = is_na(b) and st.checkbox(t("wi.levers.keep_na", name=friendly(col)), value=True, key=f"wi_{col}_na")
    default = 0.0 if is_na(b) else min(max(float(b), 0.0), mx)
    v = st.slider(f"{friendly(col)}", 0.0, float(mx), value=float(default), step=step_of(mx), key=f"wi_{col}",
                  disabled=keep_na, help=t(f"feat.{col}.desc"))
    scenario[col] = np.nan if keep_na else float(v)


st.markdown(f"**{t('wi.levers.title')}** · {t('wi.levers.sub')}")
cols = st.columns(3)
for i, c in enumerate(MAIN):
    with cols[i % 3]:
        lever(c)
with st.expander(t("wi.levers.more"), icon=":material/tune:"):
    cols = st.columns(3)
    for i, c in enumerate(MORE):
        with cols[i % 3]:
            lever(c)
st.caption(t("wi.levers.ranges_eda") if (eda or {}).get("numeric") else t("wi.levers.ranges_fallback"))

frame = pd.DataFrame([baseline, scenario])
frame[ID] = ["baseline", "scenario"]
p_base, p_scen = scoring.score_rows(frame, bundle)
delta_pp = (p_scen - p_base) * 100

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.markdown(f"**{t('wi.compare.baseline')}**")
    prob_gauge(p_base, reference=pf.churn_rate / 100)
with c2:
    st.markdown(f"**{t('wi.compare.scenario')}**")
    prob_gauge(p_scen, reference=pf.churn_rate / 100,
               color=charts.risk_down() if p_scen < p_base else (charts.risk_up() if p_scen > p_base else None))
with c3:
    changed = [c for c in LEVERS if not (is_na(baseline.get(c)) and is_na(scenario.get(c)))
               and (is_na(baseline.get(c)) != is_na(scenario.get(c)) or
                    (not is_na(baseline.get(c)) and abs(float(baseline[c]) - float(scenario[c])) > 1e-9))]
    rel = (p_scen / p_base - 1) * 100 if p_base > 0 else float("nan")
    kpi_row([
        dict(label=t("wi.compare.change"), value=fmt.pp(delta_pp, 2),
             delta=t("wi.compare.relative", rel=fmt.pct(rel, 0)) if not math.isnan(rel) else None,
             tone="good" if delta_pp < 0 else ("bad" if delta_pp > 0 else "neutral")),
        dict(label=t("wi.compare.levers_moved"), value=str(len(changed)),
             foot=", ".join(friendly(c) for c in changed) or t("wi.compare.none")),
    ], cols=1)


@st.cache_data(show_spinner=False, max_entries=64)
def sweep(base_items: tuple, ranges_items: tuple, points: int = 25) -> pd.DataFrame:
    """Individual partial-dependence curves: each lever swept alone, the rest held at the scenario."""
    base = dict(base_items)
    rng = dict(ranges_items)
    rows, meta = [], []
    for col in LEVERS:
        start = 1.0 if col == "REGULARITY" else 0.0
        for v in np.linspace(start, rng[col], points):
            r = dict(base)
            r[col] = float(v)
            rows.append(r)
            meta.append((col, float(v)))
    df = pd.DataFrame(rows)
    df[ID] = np.arange(len(df)).astype(str)
    probs = scoring.score_rows(df, artifacts.get_bundle())
    return pd.DataFrame({"lever": [m[0] for m in meta], "value": [m[1] for m in meta], "prob": probs})


def _key(d: dict) -> tuple:
    return tuple(sorted((k, None if is_na(v) else v) for k, v in d.items()))


st.markdown(f"**{t('wi.sweep.title')}**")
st.caption(t("wi.sweep.sub"))
sw = sweep(_key(scenario), tuple(sorted(ranges.items())))
best = sw.loc[sw.groupby("lever")["prob"].idxmin()].set_index("lever")
gain = ((p_scen - best["prob"]) * 100).clip(lower=0).sort_values()
gain_df = pd.DataFrame({"lever": gain.index, "label": [friendly(c) for c in gain.index], "gain": gain.values,
                        "best_value": best.loc[gain.index, "value"].values})
top_lever = gain_df.iloc[-1]
lever_helps = bool(top_lever["gain"] > 0.005)
bar = go.Figure(go.Bar(
    y=gain_df["label"], x=gain_df["gain"], orientation="h", marker_color=charts.risk_down(),
    text=[fmt.pp(-g, 2) if g > 0.005 else "0" for g in gain_df["gain"]], textposition="outside",
    customdata=gain_df[["best_value"]],
    hovertemplate="<b>%{y}</b><br>" + t("wi.sweep.hover_gain") + ": %{x:.2f} pp<br>"
                  + t("wi.sweep.hover_best") + ": %{customdata[0]:,.0f}<extra></extra>",
))
charts.style(bar, height=420, legend=False, xgrid=True, ygrid=False, title=t("wi.sweep.bar_title"))
bar.update_xaxes(title=t("wi.sweep.bar_axis"), ticksuffix=" pp", range=[0, max(gain_df["gain"].max() * 1.25, 0.5)])
so = (t("wi.sweep.so_what", name=top_lever["label"], gain=fmt.num(top_lever["gain"], 2),
        value=fmt.num(top_lever["best_value"])) if lever_helps else t("wi.sweep.so_what_none"))

c1, c2 = st.columns(2)
with c1:
    charts.show(bar, so, data=gain_df[["label", "gain", "best_value"]].iloc[::-1].rename(columns={
        "label": t("wi.sweep.col_lever"), "gain": t("wi.sweep.hover_gain"), "best_value": t("wi.sweep.hover_best")}
    ).round(2), key="jr_bar")
with c2:
    pick = st.selectbox(t("wi.sweep.pick"), list(gain_df["lever"][::-1]), format_func=friendly, key="wi_pick")
    curve = sw[sw["lever"] == pick]
    line = go.Figure()
    line.add_trace(go.Scatter(x=curve["value"], y=curve["prob"] * 100, mode="lines", name=friendly(pick),
                              line=dict(color=charts.accent(), width=2),
                              hovertemplate=f"{friendly(pick)}: %{{x:,.0f}}<br>{t('ui.gauge.label')}: %{{y:.2f}}%"
                                            "<extra></extra>"))
    cur = scenario.get(pick)
    if not is_na(cur):
        line.add_trace(go.Scatter(x=[cur], y=[p_scen * 100], mode="markers", name=t("wi.sweep.current"),
                                  marker=dict(size=11, color=charts.risk_up(),
                                              line=dict(width=2, color=charts.ink("surface"))),
                                  hovertemplate=t("wi.sweep.current") + ": %{x:,.0f} → %{y:.2f}%<extra></extra>"))
    line.add_hline(y=pf.churn_rate, line=dict(color=charts.ink("muted"), width=1),
                   annotation_text=t("exec.portfolio_line", rate=fmt.pct(pf.churn_rate, 2)),
                   annotation_font=dict(size=11, color=charts.ink("secondary")))
    charts.style(line, height=380, pct_y=True, xgrid=True, title=t("wi.sweep.curve_title", name=friendly(pick)))
    line.update_xaxes(title=friendly(pick))
    line.update_yaxes(title=t("ui.gauge.label"))
    lo_row = curve.loc[curve["prob"].idxmin()]
    charts.show(line, t("wi.sweep.curve_so_what", name=friendly(pick), value=fmt.num(lo_row["value"]),
                        p=fmt.pct(lo_row["prob"] * 100, 2)),
                data=curve[["value", "prob"]].assign(prob=lambda d: (d["prob"] * 100).round(3)).rename(columns={
                    "value": friendly(pick), "prob": t("ui.gauge.label")}), key="jr_line")
st.caption(t("wi.sweep.caveat"))

# ===========================================================================
# 5 · What should we do?
# ===========================================================================
step_header(5, t("jr.s5.title"), t("jr.s5.sub"), "jr-act")
ctx = explain.build_context(prob, res["grouped"], raw, res["features"], pf.churn_rate / 100, profiles)
actions = explain.next_best_actions(ctx)
lever_line = (t("jr.s5.lever", name=top_lever["label"], value=fmt.num(top_lever["best_value"]),
                gain=fmt.num(top_lever["gain"], 2)) if lever_helps else t("wi.sweep.so_what_none"))


def action_card() -> str:
    """Plain-text summary an agent can paste into a CRM note."""
    lines = [t("jr.card.title"), "=" * 40,
             f"{t('jr.card.customer')}: {cust_id}",
             f"{t('jr.card.source')}: {how}",
             f"{t('ui.gauge.label')}: {fmt.pct(prob * 100, 1)} ({fmt.num(mult, 2)}× {t('jr.card.portfolio')} "
             f"{fmt.pct(pf.churn_rate, 2)})",
             f"{t('cc.result.segment')}: {segment_name(seg)}", "",
             f"{t('cc.reasons.title')}:"]
    lines += [f"  {'▲' if d == 'up' else '▼'} {s}" for d, s in reasons]
    lines += ["", f"{t('cc.nba.title')}:"]
    lines += [f"  - {t(f'nba.{k}.title')}: {t(f'nba.{k}.body')}" for k in actions]
    lines += ["", f"{t('jr.s5.lever_title')}: {lever_line}", "",
              t("cc.nba.caption"), t("wi.sweep.caveat"),
              f"{t('jr.card.generated')}: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"]
    return "\n".join(lines) + "\n"


c1, c2 = st.columns([1.2, 1])
with c1:
    st.markdown(f"**{t('cc.nba.title')}**")
    st.caption(t("cc.nba.sub"))
    for key in actions:
        insight_callout(t(f"nba.{key}.body"), "good" if key == "low_risk" else "info", tag=t(f"nba.{key}.title"))
    st.caption(t("cc.nba.caption"))
with c2:
    st.markdown(f"**{t('jr.s5.lever_title')}**")
    insight_callout(lever_line, "info", tag=t("jr.s5.lever_tag"))
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", cust_id)[:40] or "customer"
    st.download_button(t("jr.s5.download"), data=action_card().encode("utf-8"),
                       file_name=f"action_card_{safe_id}.txt", mime="text/plain", on_click="ignore",
                       icon=":material/download:", key="jr_card_dl", width="stretch")
    st.markdown(f"**{t('jr.s5.next_title')}**")
    navigation.link("retention", label=t("jr.s5.to_retention"))
    navigation.link("batch", label=t("jr.s5.to_batch"))
    navigation.link("copilot", label=t("jr.s5.to_copilot"))

# ===========================================================================
# At-a-glance strip (top of the page)
# ===========================================================================
driver = res["grouped"].index[0] if len(res["grouped"]) else None
short_id = cust_id if len(cust_id) <= 16 else cust_id[:14] + "…"
strip.html(journey_strip([
    dict(n=1, title=t("jr.strip.find"), value=short_id, full=cust_id, detail=how_short, anchor="jr-find"),
    dict(n=2, title=t("jr.strip.risk"), value=fmt.pct(prob * 100, 1),
         detail=t("jr.strip.risk_detail", mult=fmt.num(mult, 2)), anchor="jr-risk"),
    dict(n=3, title=t("jr.strip.why"), value=friendly(driver) if driver else "–",
         detail=t("jr.strip.why_up") if driver and res["grouped"].iloc[0] > 0 else t("jr.strip.why_down"),
         anchor="jr-why"),
    dict(n=4, title=t("jr.strip.whatif"), value=top_lever["label"] if lever_helps else "–",
         detail=t("jr.strip.whatif_detail", gain=fmt.pp(-top_lever["gain"], 1)) if lever_helps
         else t("jr.strip.whatif_none"), anchor="jr-whatif"),
    dict(n=5, title=t("jr.strip.act"), value=t(f"nba.{actions[0]}.title") if actions else "–",
         detail=t("jr.strip.act_detail", n=len(actions)), anchor="jr-act"),
], label=t("jr.strip.label")))
