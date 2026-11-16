"""Segment Explorer: compare segments, benchmark them and pick a strategy."""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, fmt, portfolio
from app.core.segments import SEGMENT_IDS, segment_actions, segment_blurb, with_names
from app.i18n import t
from app.ui import charts
from app.ui.components import insight_callout, kpi_card, kpi_row, page_header, section_header

profiles = with_names(artifacts.get_segment_profiles())
pf = portfolio.compute(profiles, artifacts.get_metadata())
p = profiles.set_index("segment")
names = p["segment_name"].to_dict()

page_header(t("segx.title"), t("segx.sub"), t("segx.eyebrow"))

# ---------------------------------------------------------------------------
# Segment strip
# ---------------------------------------------------------------------------
cards = []
for sid in SEGMENT_IDS:
    r = p.loc[sid]
    cards.append(
        f'<div class="hub-card"><div style="display:flex;align-items:center;gap:.45rem">'
        f'<span style="width:.75rem;height:.75rem;border-radius:3px;background:{charts.segment_color(sid)}"></span>'
        f'<b>{html.escape(r["segment_name"])}</b></div>'
        f'<div style="font-size:.82rem;opacity:.75;margin:.35rem 0 .5rem 0;line-height:1.4">{html.escape(segment_blurb(sid))}</div>'
        f'<div class="hub-kpi-label">{html.escape(t("segx.strip.customers"))}</div>'
        f'<div style="font-weight:800;font-size:1.15rem">{fmt.num(r["customers"])}</div>'
        f'<div class="hub-kpi-label" style="margin-top:.3rem">{html.escape(t("segx.strip.churn"))}</div>'
        f'<div style="font-weight:800;font-size:1.15rem">{fmt.pct(r["churn_rate"], 1)}</div></div>'
    )
st.html(f'<div class="hub-grid" style="--cols:5">{"".join(cards)}</div>')

# ---------------------------------------------------------------------------
# Pick two segments
# ---------------------------------------------------------------------------
section_header(t("segx.compare.title"), t("segx.compare.sub"))
c1, c2 = st.columns(2)
with c1:
    a = st.selectbox(t("segx.compare.primary"), SEGMENT_IDS, index=0, format_func=names.__getitem__, key="segx_a")
with c2:
    b = st.selectbox(t("segx.compare.comparison"), SEGMENT_IDS, index=1, format_func=names.__getitem__, key="segx_b")


def side(sid: int) -> None:
    r = p.loc[sid]
    st.html(f'<div style="display:flex;align-items:center;gap:.5rem;margin:.4rem 0">'
            f'<span style="width:.9rem;height:.9rem;border-radius:4px;background:{charts.segment_color(sid)}"></span>'
            f'<b style="font-size:1.15rem">{html.escape(r["segment_name"])}</b></div>')
    kpi_row([
        dict(label=t("segx.m.customers"), value=fmt.num(r["customers"]),
             foot=t("segx.m.of_portfolio", share=fmt.pct(r["pct_customers"], 1))),
        dict(label=t("segx.m.churn"), value=fmt.pct(r["churn_rate"], 2)),
        dict(label=t("segx.m.revenue_share"), value=fmt.pct(r["pct_of_revenue"], 1)),
        dict(label=t("segx.m.active_days"), value=fmt.num(r["median_active_days"])),
        dict(label=t("segx.m.data_users"), value=fmt.pct(r["data_users"], 1)),
        dict(label=t("segx.m.recharges"), value=fmt.num(r["median_recharges"])),
    ], cols=2)


l, r_ = st.columns(2)
with l:
    side(a)
with r_:
    side(b)

# ---------------------------------------------------------------------------
# Radar on normalised metrics
# ---------------------------------------------------------------------------
section_header(t("segx.radar.title"), t("segx.radar.sub"))
METRICS = [("churn_rate", "segx.metric.churn"), ("median_active_days", "segx.metric.active_days"),
           ("data_users", "segx.metric.data_users"), ("pct_of_revenue", "segx.metric.revenue_share"),
           ("median_recharges", "segx.metric.recharges"), ("offnet_share", "segx.metric.offnet")]
chosen = st.multiselect(t("segx.radar.pick"), SEGMENT_IDS, default=sorted({a, b}), format_func=names.__getitem__,
                        key="segx_radar")
labels = [t(k) for _, k in METRICS]
maxima = {col: float(profiles[col].max()) or 1.0 for col, _ in METRICS}
radar = go.Figure()
for sid in chosen:
    r = p.loc[sid]
    vals = [r[col] / maxima[col] * 100 for col, _ in METRICS]
    raw = [r[col] for col, _ in METRICS]
    radar.add_trace(go.Scatterpolar(
        r=vals + vals[:1], theta=labels + labels[:1], name=r["segment_name"], fill="toself",
        line=dict(color=charts.segment_color(sid), width=2), fillcolor=charts.segment_color(sid), opacity=0.55,
        customdata=raw + raw[:1],
        hovertemplate="<b>" + r["segment_name"] + "</b><br>%{theta}: %{customdata:,.1f}<br>"
                      + t("segx.radar.hover_norm") + ": %{r:.0f}<extra></extra>",
    ))
charts.style(radar, height=440)
radar.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=10),
                                               gridcolor=charts.ink("grid"), linecolor=charts.ink("axis")),
                               angularaxis=dict(gridcolor=charts.ink("grid"), linecolor=charts.ink("axis"))))
if chosen:
    high = max(chosen, key=lambda s: p.loc[s, "median_active_days"])
    low = min(chosen, key=lambda s: p.loc[s, "median_active_days"])
    so = t("segx.radar.so_what", high=names[high], low=names[low],
           hd=fmt.num(p.loc[high, "median_active_days"]), ld=fmt.num(p.loc[low, "median_active_days"]))
    table = profiles.loc[profiles["segment"].isin(chosen), ["segment_name"] + [c for c, _ in METRICS]]
    table.columns = [t("exec.col.segment")] + labels
    charts.show(radar, so, data=table.round(2), key="segx_radar_chart")
else:
    st.info(t("segx.radar.empty"))

# ---------------------------------------------------------------------------
# Benchmarks vs portfolio
# ---------------------------------------------------------------------------
section_header(t("segx.bench.title"), t("segx.bench.sub"))


def bench_cards(sid: int) -> list[dict]:
    r = p.loc[sid]
    out = []
    specs = [
        ("segx.m.churn", r["churn_rate"], pf.churn_rate, "pct2", False),
        ("segx.m.active_days", r["median_active_days"], pf.active_days_benchmark, "num", True),
        ("segx.m.data_users", r["data_users"], pf.data_users_benchmark, "pct1", True),
        ("segx.m.recharges", r["median_recharges"], pf.recharges_benchmark, "num", True),
    ]
    for key, value, bench, kind, higher_better in specs:
        diff = value - bench
        better = (diff > 0) == higher_better
        tone = "neutral" if abs(diff) < 1e-9 else ("good" if better else "bad")
        if kind == "pct2":
            v, d = fmt.pct(value, 2), fmt.pp(diff, 2)
        elif kind == "pct1":
            v, d = fmt.pct(value, 1), fmt.pp(diff, 1)
        else:
            v, d = fmt.num(value), f"{'+' if diff >= 0 else '−'}{fmt.num(abs(diff), 1)}"
        out.append(dict(label=t(key), value=v, delta=t("segx.bench.vs", d=d), tone=tone))
    value_gap = r["pct_of_revenue"] - r["pct_customers"]
    out.append(dict(label=t("segx.bench.value_gap"), value=fmt.pp(value_gap, 1),
                    delta=t("segx.bench.value_gap_delta"), tone="good" if value_gap > 0 else "bad",
                    help=t("segx.bench.value_gap_help")))
    return out


l, r_ = st.columns(2)
with l:
    st.markdown(f"**{names[a]}**")
    kpi_row(bench_cards(a), cols=2)
with r_:
    st.markdown(f"**{names[b]}**")
    kpi_row(bench_cards(b), cols=2)
st.caption(t("segx.bench.caption", churn=fmt.pct(pf.churn_rate, 2), days=fmt.num(pf.active_days_benchmark, 1),
             data=fmt.pct(pf.data_users_benchmark, 1), rech=fmt.num(pf.recharges_benchmark, 1)))

# ---------------------------------------------------------------------------
# Management interpretation (generated from the numbers)
# ---------------------------------------------------------------------------
section_header(t("segx.interp.title"), t("segx.interp.sub"))


def interpretation(sid: int) -> list[str]:
    r = p.loc[sid]
    out = []
    churn_gap = r["churn_rate"] - pf.churn_rate
    active_gap = r["median_active_days"] - pf.active_days_benchmark
    data_gap = r["data_users"] - pf.data_users_benchmark
    rev_gap = r["pct_of_revenue"] - r["pct_customers"]
    if churn_gap > 5:
        out.append(t("segx.interp.churn_above", gap=fmt.num(churn_gap, 2)))
    elif churn_gap < -5:
        out.append(t("segx.interp.churn_below", gap=fmt.num(abs(churn_gap), 2)))
    else:
        out.append(t("segx.interp.churn_aligned", gap=fmt.pp(churn_gap, 2)))
    if active_gap < -5:
        out.append(t("segx.interp.active_below", gap=fmt.num(abs(active_gap), 1)))
    elif active_gap > 5:
        out.append(t("segx.interp.active_above", gap=fmt.num(active_gap, 1)))
    if data_gap < -10:
        out.append(t("segx.interp.data_below", gap=fmt.num(abs(data_gap), 1)))
    elif data_gap > 10:
        out.append(t("segx.interp.data_above", gap=fmt.num(data_gap, 1)))
    if rev_gap > 10:
        out.append(t("segx.interp.value_high", cust=fmt.pct(r["pct_customers"], 1), rev=fmt.pct(r["pct_of_revenue"], 1)))
    elif rev_gap < -10:
        out.append(t("segx.interp.value_low", cust=fmt.pct(r["pct_customers"], 1), rev=fmt.pct(r["pct_of_revenue"], 1)))
    return out


l, r_ = st.columns(2)
for col, sid in ((l, a), (r_, b)):
    with col:
        st.markdown(f"**{names[sid]}**")
        tone = "risk" if p.loc[sid, "churn_rate"] > pf.churn_rate + 5 else "info"
        for line in interpretation(sid):
            insight_callout(line, tone)
st.caption(t("segx.interp.caption"))

# ---------------------------------------------------------------------------
# Recommended actions (team's playbook)
# ---------------------------------------------------------------------------
section_header(t("segx.actions.title"), t("segx.actions.sub"))
l, r_ = st.columns(2)
for col, sid in ((l, a), (r_, b)):
    with col:
        with st.container(border=True):
            st.markdown(f"**{names[sid]}**")
            for i, action in enumerate(segment_actions(sid), start=1):
                st.markdown(f"{i}. {action}")
st.caption(t("segx.actions.caption"))

# ---------------------------------------------------------------------------
# Strategy matrix
# ---------------------------------------------------------------------------
section_header(t("segx.matrix.title"), t("segx.matrix.sub"))
c1, c2 = st.columns(2)
with c1:
    y_cut = st.slider(t("segx.matrix.churn_cut"), 0.0, 60.0, float(round(pf.churn_rate, 2)), 0.25,
                      help=t("segx.matrix.churn_cut_help"), key="segx_ycut")
with c2:
    fair = 100.0 / len(profiles)
    x_cut = st.slider(t("segx.matrix.rev_cut"), 0.0, 70.0, float(fair), 0.5,
                      help=t("segx.matrix.rev_cut_help"), key="segx_xcut")

QUAD = {("hi", "hi"): "protect", ("hi", "lo"): "grow", ("lo", "hi"): "reactivate", ("lo", "lo"): "deprioritise"}


def quadrant(rev: float, churn: float) -> str:
    return QUAD[("hi" if rev >= x_cut else "lo", "hi" if churn >= y_cut else "lo")]


xmax, ymax = 70.0, 60.0
mx = go.Figure()
shade = {"protect": charts.risk_up(), "grow": charts.series(5), "reactivate": charts.series(3),
         "deprioritise": charts.neutral()}
for (xs, ys), q in QUAD.items():
    x0, x1 = (x_cut, xmax) if xs == "hi" else (0, x_cut)
    y0, y1 = (y_cut, ymax) if ys == "hi" else (0, y_cut)
    mx.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=shade[q], opacity=0.07, line_width=0,
                 layer="below")
    mx.add_annotation(x=x0 + 1 if xs == "lo" else x1 - 1, y=y1 - 2, text=f"<b>{t(f'segx.quad.{q}.name')}</b>",
                      showarrow=False, xanchor="left" if xs == "lo" else "right", yanchor="top",
                      font=dict(size=13, color=charts.ink("secondary")))
mx.add_vline(x=x_cut, line=dict(color=charts.ink("axis"), width=1))
mx.add_hline(y=y_cut, line=dict(color=charts.ink("axis"), width=1))
for sid in SEGMENT_IDS:
    r = p.loc[sid]
    mx.add_trace(go.Scatter(
        x=[r["pct_of_revenue"]], y=[r["churn_rate"]], mode="markers+text", name=r["segment_name"],
        text=[r["segment_name"]], textposition="middle right" if r["pct_of_revenue"] < 50 else "middle left",
        textfont=dict(color=charts.ink("primary"), size=12),
        marker=dict(size=16, color=charts.segment_color(sid), line=dict(width=2, color=charts.ink("surface"))),
        hovertemplate=f"<b>{r['segment_name']}</b><br>{t('segx.metric.revenue_share')}: %{{x:.1f}}%<br>"
                      f"{t('segx.metric.churn')}: %{{y:.2f}}%<extra></extra>",
    ))
charts.style(mx, height=460, pct_x=True, pct_y=True, xgrid=True)
mx.update_xaxes(title=t("segx.metric.revenue_share"), range=[-1, xmax])
mx.update_yaxes(title=t("segx.metric.churn"), range=[-2, ymax])

assign = pd.DataFrame({
    "segment": [names[s] for s in SEGMENT_IDS],
    "quadrant": [t(f"segx.quad.{quadrant(p.loc[s, 'pct_of_revenue'], p.loc[s, 'churn_rate'])}.name")
                 for s in SEGMENT_IDS],
    "revenue": [round(p.loc[s, "pct_of_revenue"], 2) for s in SEGMENT_IDS],
    "churn": [round(p.loc[s, "churn_rate"], 2) for s in SEGMENT_IDS],
})
protect = [names[s] for s in SEGMENT_IDS if quadrant(p.loc[s, "pct_of_revenue"], p.loc[s, "churn_rate"]) == "protect"]
react = [names[s] for s in SEGMENT_IDS if quadrant(p.loc[s, "pct_of_revenue"], p.loc[s, "churn_rate"]) == "reactivate"]
if protect:
    so = t("segx.matrix.so_what_protect", segs=", ".join(protect))
else:
    safest_value = names[int(p["pct_of_revenue"].idxmax())]
    so = t("segx.matrix.so_what_none", seg=safest_value, react=", ".join(react) or "–")
charts.show(mx, so, data=assign.rename(columns={"segment": t("exec.col.segment"), "quadrant": t("segx.matrix.col_quad"),
                                                "revenue": t("segx.metric.revenue_share"),
                                                "churn": t("segx.metric.churn")}), key="segx_matrix")
qcols = st.columns(4)
for col, q in zip(qcols, ("protect", "grow", "reactivate", "deprioritise")):
    with col:
        kpi_card(t(f"segx.quad.{q}.name"), ", ".join(assign.loc[assign["quadrant"] == t(f"segx.quad.{q}.name"),
                                                                   "segment"]) or "–",
                 foot=t(f"segx.quad.{q}.desc"))
