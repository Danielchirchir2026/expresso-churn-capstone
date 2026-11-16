"""Executive Dashboard: portfolio KPIs, segment economics, computed insights."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import navigation
from app.core import artifacts, fmt, portfolio
from app.core import state
from app.core.features import friendly, global_importance, grouped_importance
from app.core.pdf import executive_brief
from app.core.segments import with_names
from app.i18n import current_lang, t
from app.ui import charts
from app.ui.components import insight_callout, kpi_row, page_header, section_header

profiles = with_names(artifacts.get_segment_profiles())
meta = artifacts.get_metadata()
config = artifacts.get_config()
pf = portfolio.compute(profiles, meta)
p = profiles.set_index("segment")

page_header(t("exec.title"), t("exec.sub"), t("exec.eyebrow"))
_lang = current_lang()
_raw_profiles = artifacts.get_segment_profiles()
_batch = state.get_batch()  # captured now: the deferred callable runs outside the script context
st.download_button(t("exec.brief_btn"), data=lambda: executive_brief(_raw_profiles, meta, _lang, _batch),
                   file_name=f"expresso_executive_brief_{_lang}.pdf", mime="application/pdf",
                   icon=":material/picture_as_pdf:", on_click="ignore", key="exec_pdf", help=t("exec.brief_help"))

kpi_row([
    dict(label=t("kpi.total_customers"), value=fmt.num(pf.total_customers), help=t("kpi.total_customers_help")),
    dict(label=t("kpi.churn_rate"), value=fmt.pct(pf.churn_rate, 2), help=t("kpi.churn_rate_help")),
    dict(label=t("kpi.est_churners"), value=fmt.num(pf.est_churners), help=t("kpi.est_churners_help")),
    dict(label=t("kpi.roc_auc"), value=fmt.num(pf.roc_auc, 4), help=t("kpi.roc_auc_help")),
    dict(label=t("kpi.revenue_concentration"), value=fmt.pct(pf.revenue_concentration, 1),
         help=t("kpi.revenue_concentration_help"),
         foot=t("exec.kpi_foot.customers_share", share=fmt.pct(p.loc[1, "pct_customers"], 1))),
    dict(label=t("kpi.churners_silent"), value=fmt.pct(pf.churners_from_silent, 1),
         help=t("kpi.churners_silent_help"),
         foot=t("exec.kpi_foot.customers_share", share=fmt.pct(p.loc[0, "pct_customers"], 1))),
], cols=3)

# ---------------------------------------------------------------------------
# Hero: bubble chart (x = % customers, y = churn rate, size = % revenue)
# ---------------------------------------------------------------------------
section_header(t("exec.bubble.title"), t("exec.bubble.sub"))

LABEL_POS = {0: "bottom center", 1: "middle left", 2: "top center", 3: "top right", 4: "top center"}
max_rev = float(profiles["pct_of_revenue"].max())
fig = go.Figure()
for _, r in profiles.iterrows():
    fig.add_trace(go.Scatter(
        x=[r["pct_customers"]], y=[r["churn_rate"]], mode="markers+text", name=r["segment_name"],
        text=[r["segment_name"]], textposition=LABEL_POS.get(int(r["segment"]), "top center"),
        textfont=dict(color=charts.ink("primary"), size=12),
        marker=dict(size=[max(r["pct_of_revenue"], 1e-3)], sizemode="area", sizeref=2.0 * max_rev / (95 ** 2),
                    sizemin=9, color=charts.segment_color(r["segment"]), opacity=0.85,
                    line=dict(width=2, color=charts.ink("surface"))),
        customdata=[[r["customers"], r["pct_of_revenue"], r["pct_of_churners"]]],
        hovertemplate=(f"<b>{r['segment_name']}</b><br>{t('exec.axis.pct_customers')}: %{{x:.1f}}%<br>"
                       f"{t('exec.axis.churn_rate')}: %{{y:.2f}}%<br>{t('exec.hover.revenue_share')}: %{{customdata[1]:.1f}}%"
                       f"<br>{t('exec.hover.churner_share')}: %{{customdata[2]:.1f}}%<br>"
                       f"{t('exec.hover.customers')}: %{{customdata[0]:,}}<extra></extra>"),
    ))
fig.add_hline(y=pf.churn_rate, line=dict(color=charts.ink("muted"), width=1),
              annotation_text=t("exec.portfolio_line", rate=fmt.pct(pf.churn_rate, 2)),
              annotation_position="top left", annotation_font=dict(color=charts.ink("secondary"), size=11))
charts.style(fig, height=470, pct_x=True, pct_y=True, xgrid=True)
fig.update_layout(legend=dict(itemsizing="constant"))
fig.update_xaxes(title=t("exec.axis.pct_customers"), range=[12, 27])
fig.update_yaxes(title=t("exec.axis.churn_rate"), range=[-9, 64])
worst = p["churn_rate"].idxmax()
top_rev = p["pct_of_revenue"].idxmax()
charts.show(
    fig,
    t("exec.bubble.so_what", worst=p.loc[worst, "segment_name"], worst_rate=fmt.pct(p.loc[worst, "churn_rate"], 1),
      worst_rev=fmt.pct(p.loc[worst, "pct_of_revenue"], 1), top=p.loc[top_rev, "segment_name"],
      top_rev=fmt.pct(p.loc[top_rev, "pct_of_revenue"], 0), top_rate=fmt.pct(p.loc[top_rev, "churn_rate"], 2)),
    data=profiles[["segment_name", "pct_customers", "churn_rate", "pct_of_revenue"]].rename(columns={
        "segment_name": t("exec.col.segment"), "pct_customers": t("exec.axis.pct_customers"),
        "churn_rate": t("exec.axis.churn_rate"), "pct_of_revenue": t("exec.hover.revenue_share")}).round(2),
    key="exec_bubble",
)

# ---------------------------------------------------------------------------
# Treemap + churners vs revenue
# ---------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    section_header(t("exec.treemap.title"), t("exec.treemap.sub"))
    tm = go.Figure(go.Treemap(
        labels=profiles["segment_name"], parents=[""] * len(profiles), values=profiles["customers"],
        marker=dict(colors=profiles["churn_rate"], colorscale=charts.sequential(), cmin=0,
                    colorbar=dict(title=dict(text=t("exec.treemap.colorbar")), ticksuffix="%", thickness=10),
                    line=dict(width=2, color=charts.ink("surface")), pad=dict(t=2, l=2, r=2, b=2)),
        customdata=profiles["churn_rate"],
        texttemplate="<b>%{label}</b><br>%{value:,}<br>%{customdata:.1f}% " + t("exec.treemap.churn_word"),
        hovertemplate="<b>%{label}</b><br>" + t("exec.hover.customers") + ": %{value:,}<br>"
                      + t("exec.axis.churn_rate") + ": %{customdata:.2f}%<extra></extra>",
        tiling=dict(pad=2), sort=True,
    ))
    charts.style(tm, height=380, legend=False)
    tm.update_layout(margin=dict(l=4, r=4, t=8, b=4))
    biggest = p["customers"].idxmax()
    charts.show(
        tm,
        t("exec.treemap.so_what", big=p.loc[biggest, "segment_name"],
          share=fmt.pct(p.loc[biggest, "pct_customers"], 1), rate=fmt.pct(p.loc[biggest, "churn_rate"], 1)),
        data=profiles[["segment_name", "customers", "churn_rate"]].rename(columns={
            "segment_name": t("exec.col.segment"), "customers": t("exec.hover.customers"),
            "churn_rate": t("exec.axis.churn_rate")}).round(2),
        key="exec_treemap",
    )

with right:
    section_header(t("exec.share.title"), t("exec.share.sub"))
    order = profiles.sort_values("pct_of_churners")
    bars = go.Figure()
    bars.add_trace(go.Bar(y=order["segment_name"], x=order["pct_of_churners"], orientation="h",
                          name=t("exec.share.churners"), marker_color=charts.risk_up(),
                          text=[fmt.pct(v, 1) for v in order["pct_of_churners"]], textposition="outside",
                          hovertemplate="%{y}: %{x:.1f}%<extra>" + t("exec.share.churners") + "</extra>"))
    bars.add_trace(go.Bar(y=order["segment_name"], x=order["pct_of_revenue"], orientation="h",
                          name=t("exec.share.revenue"), marker_color=charts.series(6),
                          text=[fmt.pct(v, 1) for v in order["pct_of_revenue"]], textposition="outside",
                          hovertemplate="%{y}: %{x:.1f}%<extra>" + t("exec.share.revenue") + "</extra>"))
    charts.style(bars, height=380, pct_x=True, xgrid=True, ygrid=False)
    bars.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    bars.update_xaxes(range=[0, max(order["pct_of_churners"].max(), order["pct_of_revenue"].max()) * 1.22])
    charts.show(
        bars,
        t("exec.share.so_what", seg=p.loc[worst, "segment_name"], churners=fmt.pct(p.loc[worst, "pct_of_churners"], 0),
          rev=fmt.pct(p.loc[worst, "pct_of_revenue"], 1)),
        data=profiles[["segment_name", "pct_of_churners", "pct_of_revenue"]].rename(columns={
            "segment_name": t("exec.col.segment"), "pct_of_churners": t("exec.share.churners"),
            "pct_of_revenue": t("exec.share.revenue")}).round(2),
        key="exec_share",
    )

# ---------------------------------------------------------------------------
# Executive insights (computed)
# ---------------------------------------------------------------------------
section_header(t("exec.insights.title"), t("exec.insights.sub"))
for tone, text in portfolio.executive_insights(profiles, pf):
    insight_callout(text, tone)

# ---------------------------------------------------------------------------
# What drives churn (model gain importance grouped by raw column)
# ---------------------------------------------------------------------------
section_header(t("exec.drivers.title"), t("exec.drivers.sub"))
imp = global_importance(artifacts.get_model(), tuple(config["feature_columns"]), tuple(config["raw_source_columns"]))
grp = grouped_importance(imp).head(8).iloc[::-1]


def driver_label(group: str) -> str:
    """Flag groups whose gain comes mostly from a 'value is missing' indicator."""
    members = imp[imp["group"] == group]
    top = members.iloc[0]
    is_missing_flag = top["feature"].endswith("_MISSING") or top["feature"].startswith("is_missing_")
    if is_missing_flag and top["gain"] > 0.5 * members["gain"].sum():
        return t("exec.drivers.mostly_missing", name=friendly(group))
    return friendly(group)


grp["label"] = grp["group"].map(driver_label)
dfig = go.Figure(go.Bar(
    y=grp["label"], x=grp["gain_share"], orientation="h", marker_color=charts.accent(),
    text=[fmt.pct(v, 1) for v in grp["gain_share"]], textposition="outside",
    customdata=grp[["group", "n_features"]],
    hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>" + t("exec.drivers.hover") + ": %{x:.1f}%<br>"
                  + t("exec.drivers.n_features") + ": %{customdata[1]}<extra></extra>",
))
charts.style(dfig, height=340, legend=False, pct_x=True, xgrid=True, ygrid=False)
dfig.update_xaxes(title=t("exec.drivers.axis"), range=[0, grp["gain_share"].max() * 1.18])
top2 = grp.iloc[::-1].head(2)
charts.show(
    dfig,
    t("exec.drivers.so_what", a=top2.iloc[0]["label"], b=top2.iloc[1]["label"],
      share=fmt.pct(top2["gain_share"].sum(), 0)),
    data=grp.iloc[::-1][["label", "group", "gain_share", "n_features"]].rename(columns={
        "label": t("exec.drivers.col_driver"), "group": t("exec.drivers.col_source"),
        "gain_share": t("exec.drivers.hover"), "n_features": t("exec.drivers.n_features")}).round(2),
    key="exec_drivers",
)
navigation.link("model", label=t("exec.drivers.more"))

# ---------------------------------------------------------------------------
# Segment management summary
# ---------------------------------------------------------------------------
section_header(t("exec.table.title"), t("exec.table.sub"))
summary = profiles[["segment_name", "customers", "pct_customers", "churn_rate", "pct_of_churners", "pct_of_revenue",
                    "median_revenue", "median_active_days"]]
st.dataframe(
    summary,
    hide_index=True,
    width="stretch",
    column_config={
        "segment_name": st.column_config.TextColumn(t("exec.col.segment")),
        "customers": st.column_config.NumberColumn(t("exec.hover.customers"), format="localized"),
        "pct_customers": st.column_config.NumberColumn(t("exec.axis.pct_customers"), format="%.1f%%"),
        "churn_rate": st.column_config.ProgressColumn(t("exec.axis.churn_rate"), format="%.2f%%", min_value=0,
                                                      max_value=100),
        "pct_of_churners": st.column_config.NumberColumn(t("exec.share.churners"), format="%.1f%%"),
        "pct_of_revenue": st.column_config.NumberColumn(t("exec.share.revenue"), format="%.1f%%"),
        "median_revenue": st.column_config.NumberColumn(t("exec.col.median_revenue"), format="localized"),
        "median_active_days": st.column_config.NumberColumn(t("exec.col.median_days"), format="%d"),
    },
)
st.caption(t("exec.provenance"))
