"""What-If Simulator: move behaviour levers and watch churn probability move."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import navigation
from app.core import artifacts, fmt, portfolio, presets, scoring, state
from app.core.features import friendly
from app.core.segments import SEGMENT_IDS, segment_name
from app.i18n import t
from app.ui import charts
from app.ui.components import kpi_row, page_header, prob_gauge, section_header

bundle = artifacts.get_bundle()
config = bundle["preprocessing_config"]
profiles = artifacts.get_segment_profiles()
pf = portfolio.compute(profiles, artifacts.get_metadata())
eda = artifacts.get_optional("eda_summary")

MAIN = ["REGULARITY", "FREQUENCE_RECH", "MONTANT", "REVENUE", "DATA_VOLUME", "ON_NET"]
MORE = ["ARPU_SEGMENT", "FREQUENCE", "ORANGE", "TIGO", "FREQ_TOP_PACK"]
LEVERS = MAIN + MORE
SCALE_OF = {"MONTANT": "REVENUE", "ARPU_SEGMENT": "REVENUE", "FREQUENCE": "FREQUENCE_RECH",
            "FREQ_TOP_PACK": "FREQUENCE_RECH"}

page_header(t("wi.title"), t("wi.sub"), t("wi.eyebrow"))


def is_na(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------------------
# Choose the starting customer
# ---------------------------------------------------------------------------
last = state.get_last_customer()
options = (["last"] if last else []) + [f"seg{s}" for s in SEGMENT_IDS]
labels = {"last": t("wi.source.last", id=(last or {}).get("user_id", ""))}
labels.update({f"seg{s}": t("wi.source.preset", name=segment_name(s)) for s in SEGMENT_IDS})


def reset_levers() -> None:
    for c in LEVERS:
        st.session_state.pop(f"wi_{c}", None)
        st.session_state.pop(f"wi_{c}_na", None)


source = st.selectbox(t("wi.source.label"), options, format_func=labels.__getitem__, key="wi_source",
                      on_change=reset_levers)
if source == "last":
    baseline = dict(last)
else:
    vals, _ = presets.preset(int(source[3:]), profiles, eda)
    baseline = {"user_id": "WHAT-IF", "REGION": "DAKAR", "TENURE": config["tenure_order"][-1], "MRG": "NO",
                "TOP_PACK": np.nan, **vals}
if not last:
    st.caption(t("wi.source.no_last"))
signature = repr(sorted((k, str(v)) for k, v in baseline.items()))
if st.session_state.get("wi_signature") != signature:
    reset_levers()
    st.session_state["wi_signature"] = signature


# ---------------------------------------------------------------------------
# Slider ranges: training distribution when exported, else scale with the customer
# ---------------------------------------------------------------------------
def lever_max(col: str) -> float:
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

# ---------------------------------------------------------------------------
# Levers
# ---------------------------------------------------------------------------
section_header(t("wi.levers.title"), t("wi.levers.sub"))
scenario = dict(baseline)


def lever(col: str) -> None:
    b = baseline.get(col)
    mx = ranges[col]
    if col == "REGULARITY":
        default = int(0 if is_na(b) else min(max(round(float(b)), 0), 90))
        v = st.slider(f"{friendly(col)}", 0, 90, value=default, key=f"wi_{col}", help=t(f"feat.{col}.desc"))
        scenario[col] = float(v)
        return
    na_default = is_na(b)
    if na_default:
        keep_na = st.checkbox(t("wi.levers.keep_na", name=friendly(col)), value=True, key=f"wi_{col}_na")
    else:
        keep_na = False
    default = 0.0 if na_default else min(float(b), mx)
    v = st.slider(f"{friendly(col)}", 0.0, float(mx), value=float(default), step=step_of(mx), key=f"wi_{col}",
                  disabled=keep_na, help=t(f"feat.{col}.desc"))
    scenario[col] = np.nan if keep_na else float(v)


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

# ---------------------------------------------------------------------------
# Baseline vs scenario
# ---------------------------------------------------------------------------
frame = pd.DataFrame([baseline, scenario])
frame["user_id"] = ["baseline", "scenario"]
p_base, p_scen = scoring.score_rows(frame, bundle)
delta_pp = (p_scen - p_base) * 100

section_header(t("wi.compare.title"), t("wi.compare.sub"))
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

# ---------------------------------------------------------------------------
# Which lever helps most? (individual sensitivity sweep, cached)
# ---------------------------------------------------------------------------
section_header(t("wi.sweep.title"), t("wi.sweep.sub"))


@st.cache_data(show_spinner=False, max_entries=64)
def sweep(base_items: tuple, ranges_items: tuple, points: int = 25) -> pd.DataFrame:
    base = dict(base_items)
    rng = dict(ranges_items)
    rows, meta = [], []
    for col in LEVERS:
        lo = 1.0 if col == "REGULARITY" else 0.0
        for v in np.linspace(lo, rng[col], points):
            r = dict(base)
            r[col] = float(v)
            rows.append(r)
            meta.append((col, float(v)))
    df = pd.DataFrame(rows)
    df["user_id"] = np.arange(len(df)).astype(str)
    probs = scoring.score_rows(df, artifacts.get_bundle())
    return pd.DataFrame({"lever": [m[0] for m in meta], "value": [m[1] for m in meta], "prob": probs})


def _key(d: dict) -> tuple:
    return tuple(sorted((k, None if is_na(v) else v) for k, v in d.items()))


sw = sweep(_key(scenario), tuple(sorted(ranges.items())))
best = sw.loc[sw.groupby("lever")["prob"].idxmin()].set_index("lever")
gain = ((p_scen - best["prob"]) * 100).clip(lower=0).sort_values()
gain_df = pd.DataFrame({"lever": gain.index, "label": [friendly(c) for c in gain.index], "gain": gain.values,
                        "best_value": best.loc[gain.index, "value"].values})
bar = go.Figure(go.Bar(
    y=gain_df["label"], x=gain_df["gain"], orientation="h", marker_color=charts.risk_down(),
    text=[fmt.pp(-g, 2) if g > 0.005 else "0" for g in gain_df["gain"]], textposition="outside",
    customdata=gain_df[["best_value"]],
    hovertemplate="<b>%{y}</b><br>" + t("wi.sweep.hover_gain") + ": %{x:.2f} pp<br>"
                  + t("wi.sweep.hover_best") + ": %{customdata[0]:,.0f}<extra></extra>",
))
charts.style(bar, height=420, legend=False, xgrid=True, ygrid=False, title=t("wi.sweep.bar_title"))
bar.update_xaxes(title=t("wi.sweep.bar_axis"), ticksuffix=" pp", range=[0, max(gain_df["gain"].max() * 1.25, 0.5)])
top_lever = gain_df.iloc[-1]
so = (t("wi.sweep.so_what", name=top_lever["label"], gain=fmt.num(top_lever["gain"], 2),
        value=fmt.num(top_lever["best_value"])) if top_lever["gain"] > 0.005 else t("wi.sweep.so_what_none"))

l, r = st.columns(2)
with l:
    charts.show(bar, so, data=gain_df[["label", "gain", "best_value"]].iloc[::-1].rename(columns={
        "label": t("wi.sweep.col_lever"), "gain": t("wi.sweep.hover_gain"), "best_value": t("wi.sweep.hover_best")}
    ).round(2), key="wi_bar")
with r:
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
                                  marker=dict(size=11, color=charts.risk_up(), line=dict(width=2,
                                                                                          color=charts.ink("surface"))),
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
                    "value": friendly(pick), "prob": t("ui.gauge.label")}), key="wi_line")
st.caption(t("wi.sweep.caveat"))
navigation.link("customer_check", label=t("wi.back_to_cc"))
