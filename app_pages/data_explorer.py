"""Data Explorer: what the training data looks like and how churn varies across it."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, fmt
from app.core.features import friendly, global_importance, grouped_importance
from app.i18n import t
from app.ui import charts
from app.ui.components import insight_callout, kpi_row, page_header, section_header
from app.ui.guards import require_artifact

page_header(t("eda.title"), t("eda.sub"), t("eda.eyebrow"))
summary = require_artifact("eda_summary", "eda.missing_detail")
breakdowns = artifacts.get_optional("eda_breakdowns")

DIMENSIONS = ["TENURE", "REGION", "REGULARITY", "MRG", "TOP_PACK", "MONTANT_DECILE", "REVENUE_DECILE",
              "DATA_VOLUME_DECILE", "ON_NET_DECILE"]


def dim_label(d: str) -> str:
    if d.endswith("_DECILE"):
        return t("eda.by.decile", name=friendly(d[:-len("_DECILE")]))
    return t("eda.by.regularity") if d == "REGULARITY" else friendly(d)


if summary:
    rate = summary["churn_rate"] * 100
    miss = pd.Series(summary["missing_pct"]).drop(labels=["CHURN"], errors="ignore").sort_values(ascending=False)
    kpi_row([
        dict(label=t("eda.kpi.rows"), value=fmt.num(summary["rows"])),
        dict(label=t("eda.kpi.cols"), value=fmt.num(summary["columns"])),
        dict(label=t("eda.kpi.rate"), value=fmt.pct(rate, 2), foot=t("eda.kpi.churners", n=fmt.num(summary["churners"]))),
        dict(label=t("eda.kpi.missing"), value=fmt.pct(miss.mean(), 1), help=t("eda.kpi.missing_help")),
    ], cols=4)

    # -------------------------------------------------------------- missing
    section_header(t("eda.miss.title"), t("eda.miss.sub"))
    m = miss[miss > 0].iloc[::-1]
    fig = go.Figure(go.Bar(y=[friendly(c) for c in m.index], x=m.values, orientation="h", marker_color=charts.accent(),
                           text=[fmt.pct(v, 1) for v in m.values], textposition="outside", customdata=list(m.index),
                           hovertemplate="%{y} (%{customdata}): %{x:.1f}%<extra></extra>"))
    charts.style(fig, height=max(300, 26 * len(m) + 60), legend=False, pct_x=True, xgrid=True, ygrid=False)
    fig.update_xaxes(range=[0, 105], title=t("eda.miss.axis"))
    imp = grouped_importance(global_importance(artifacts.get_model(), tuple(artifacts.get_config()["feature_columns"]),
                                               tuple(artifacts.get_config()["raw_source_columns"])))
    charts.show(fig, t("eda.miss.so_what", top=friendly(m.index[-1]), pct=fmt.pct(m.iloc[-1], 0)),
                data=pd.DataFrame({t("eda.col.column"): m.index[::-1], t("eda.miss.axis"): m.values[::-1].round(2)}),
                key="eda_missing")
    insight_callout(t("eda.miss.why", driver=friendly(imp.iloc[0]["group"]),
                      share=fmt.pct(imp.iloc[0]["gain_share"], 0)), "info", tag=t("eda.miss.why_tag"))

    # ----------------------------------------------------------- churn by
    section_header(t("eda.by.title"), t("eda.by.sub"))
    if breakdowns is None:
        require_artifact("eda_breakdowns", "eda.missing_detail")
    else:
        dims = [d for d in DIMENSIONS if d in set(breakdowns["dimension"])]
        dim = st.selectbox(t("eda.by.pick"), dims, format_func=dim_label, key="eda_dim")
        b = breakdowns[breakdowns["dimension"] == dim].sort_values("order")
        if dim in ("REGION", "TOP_PACK"):
            b = b.sort_values("churn_rate")
        horizontal = dim in ("REGION", "TOP_PACK")
        rates = b["churn_rate"] * 100
        labels = [t("cc.form.not_recorded") if lv == "Not recorded" else lv for lv in b["level"]]
        bar = go.Bar(
            x=rates if horizontal else labels, y=labels if horizontal else rates,
            orientation="h" if horizontal else "v", marker_color=charts.accent(),
            customdata=np.stack([b["count"], b["churners"]], axis=1),
            hovertemplate=("%{y}" if horizontal else "%{x}") + ": %{" + ("x" if horizontal else "y") + ":.2f}%<br>"
                          + t("eda.by.hover") + "<extra></extra>")
        fig = go.Figure(bar)
        if horizontal:
            fig.add_vline(x=rate, line=dict(color=charts.ink("muted"), width=1))
        else:
            fig.add_hline(y=rate, line=dict(color=charts.ink("muted"), width=1),
                          annotation_text=t("eda.by.overall", rate=fmt.pct(rate, 2)),
                          annotation_font=dict(size=11, color=charts.ink("secondary")))
        charts.style(fig, height=max(360, 24 * len(b) + 80) if horizontal else 380, legend=False,
                     pct_x=horizontal, pct_y=not horizontal, xgrid=horizontal, ygrid=not horizontal)
        hi, lo = b.loc[b["churn_rate"].idxmax()], b.loc[b["churn_rate"].idxmin()]
        charts.show(fig, t("eda.by.so_what", hi=hi["level"], hr=fmt.pct(hi["churn_rate"] * 100, 1), lo=lo["level"],
                           lr=fmt.pct(lo["churn_rate"] * 100, 1), n=fmt.num(hi["count"])),
                    data=b[["level", "count", "churners", "churn_rate"]].assign(
                        churn_rate=lambda d: (d["churn_rate"] * 100).round(2)), key="eda_by")

    # -------------------------------------------------------- distributions
    section_header(t("eda.dist.title"), t("eda.dist.sub"))
    num_cols = [c for c, v in summary["numeric"].items() if v["churn_0"].get("count") and v["churn_1"].get("count")]
    c1, c2 = st.columns([2, 1])
    with c1:
        col = st.selectbox(t("eda.dist.pick"), num_cols, format_func=lambda c: f"{friendly(c)} ({c})", key="eda_dist")
    s0, s1 = summary["numeric"][col]["churn_0"], summary["numeric"][col]["churn_1"]
    can_log = min(s0["p01"], s1["p01"]) > 0  # a log axis cannot show zeros
    with c2:
        log_axis = st.toggle(t("eda.dist.log"), value=False, key="eda_log", disabled=not can_log,
                             help=None if can_log else t("eda.dist.log_disabled")) and can_log
    fig = go.Figure()
    for stats_, name, color in ((s0, t("eda.dist.stayed"), charts.risk_down()), (s1, t("eda.dist.churned"), charts.risk_up())):
        fig.add_trace(go.Box(name=name, q1=[stats_["q1"]], median=[stats_["median"]], q3=[stats_["q3"]],
                             lowerfence=[stats_["p01"]], upperfence=[stats_["p99"]], mean=[stats_["mean"]],
                             marker_color=color, line=dict(width=2), boxmean=True, x=[name]))
    charts.style(fig, height=380, legend=False)
    if log_axis:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title=friendly(col))
    charts.show(fig, t("eda.dist.so_what", name=friendly(col), m1=fmt.num(s1["median"]), m0=fmt.num(s0["median"])),
                data=pd.DataFrame([{**{"group": t("eda.dist.stayed")}, **s0}, {**{"group": t("eda.dist.churned")}, **s1}]),
                key="eda_box")
    st.caption(t("eda.dist.caption"))

    # ------------------------------------------------------------- findings
    section_header(t("eda.find.title"), t("eda.find.sub"))
    findings = [("info", t("eda.find.rate", rate=fmt.pct(rate, 2), n=fmt.num(summary["churners"]),
                           rows=fmt.num(summary["rows"])))]
    findings.append(("warn", t("eda.find.missing", col=friendly(miss.index[0]), pct=fmt.pct(miss.iloc[0], 0),
                               n=int((miss > 30).sum()))))
    r = summary["numeric"].get("REGULARITY", {})
    if r.get("churn_1", {}).get("count"):
        findings.append(("risk", t("eda.find.regularity", c=fmt.num(r["churn_1"]["median"]),
                                   s=fmt.num(r["churn_0"]["median"]))))
    if breakdowns is not None:
        for dim, key in (("TENURE", "eda.find.tenure"), ("REGION", "eda.find.region")):
            b = breakdowns[breakdowns["dimension"] == dim]
            if len(b):
                hi = b.loc[b["churn_rate"].idxmax()]
                findings.append(("info", t(key, level=hi["level"], rate=fmt.pct(hi["churn_rate"] * 100, 1),
                                           n=fmt.num(hi["count"]))))
        b = breakdowns[breakdowns["dimension"] == "REVENUE_DECILE"].sort_values("order")
        if len(b) > 2:
            first = b[b["level"].str.startswith("D1 ")]
            last = b[b["level"].str.startswith("D10")]
            if len(first) and len(last):
                findings.append(("good", t("eda.find.revenue", low=fmt.pct(first.iloc[0]["churn_rate"] * 100, 1),
                                           high=fmt.pct(last.iloc[0]["churn_rate"] * 100, 1))))
    for tone, text in findings[:6]:
        insight_callout(text, tone)
    st.caption(t("eda.source", file=summary.get("source", "Train.csv")))
