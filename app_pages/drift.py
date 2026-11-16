"""Data Quality & Drift: does the scored batch look like the training data?"""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, drift, fmt
from app.core.features import friendly
from app.i18n import t
from app.ui import charts
from app.ui.batchui import require_batch
from app.ui.components import insight_callout, kpi_row, page_header, section_header
from app.ui.guards import require_artifact

page_header(t("drift.title"), t("drift.sub"), t("drift.eyebrow"))
reference = require_artifact("reference_stats", "drift.missing_detail")
batch = require_batch("drift") if reference is not None else None

if reference is not None and batch:
    df = batch["df"]
    rep = batch.get("drift")
    if not rep or rep.get("batch_rows") != len(df):
        rep = drift.report(df, reference, artifacts.get_config())
    rows = pd.DataFrame(rep["rows"])
    counts = rep["counts"]

    verdict = {"green": "good", "amber": "warn", "red": "risk"}[rep["overall"]]
    kpi_row([
        dict(label=t("drift.kpi.checked"), value=str(len(rows)), foot=t("drift.kpi.ref_rows", n=fmt.num(rep["reference_rows"] or 0))),
        dict(label=t("drift.kpi.green"), value=str(counts.get("green", 0)), foot=t("drift.legend.green")),
        dict(label=t("drift.kpi.amber"), value=str(counts.get("amber", 0)), foot=t("drift.legend.amber")),
        dict(label=t("drift.kpi.red"), value=str(counts.get("red", 0)), foot=t("drift.legend.red")),
    ], cols=4)
    insight_callout(t(f"drift.overall.{rep['overall']}"), verdict, tag=t("drift.overall.tag"))

    # ------------------------------------------------------- traffic lights
    section_header(t("drift.table.title"), t("drift.table.sub"))
    icon = {"green": "●", "amber": "▲", "red": "■", "na": "○"}
    color = {"green": charts.STATUS["good"], "amber": charts.STATUS["warning"], "red": charts.STATUS["critical"],
             "na": charts.ink("muted")}
    order = rows.assign(rank=rows["status"].map({"red": 0, "amber": 1, "green": 2, "na": 3})).sort_values(["rank", "psi"],
                                                                                                         ascending=[True, False])
    body = "".join(
        f"<tr><td><b>{html.escape(friendly(r.column))}</b><br><span style='opacity:.6;font-size:.8rem'>{r.column}</span></td>"
        f"<td style='font-variant-numeric:tabular-nums'>{r.psi:.3f}</td>"
        f"<td><span style='color:{color[r.status]};font-weight:700'>{icon[r.status]} {html.escape(t(f'drift.status.{r.status}'))}</span></td>"
        f"<td>{fmt.pct(r.missing_ref * 100, 1)} → {fmt.pct(r.missing_new * 100, 1)}"
        f" <span style='opacity:.7'>({fmt.pp((r.missing_new - r.missing_ref) * 100, 1)})</span></td>"
        f"<td>{html.escape(', '.join(r.unseen)) if r.unseen else '–'}</td></tr>"
        for r in order.itertuples())
    head = "".join(f"<th>{html.escape(t(f'drift.col.{c}'))}</th>" for c in ("column", "psi", "status", "missing", "unseen"))
    st.html(f'<table class="hub-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')
    st.caption(t("drift.table.caption"))

    # --------------------------------------------------------- warnings
    flagged = order[order["status"].isin(["red", "amber"])]
    if len(flagged):
        section_header(t("drift.warn.title"))
        for r in flagged.itertuples():
            insight_callout(t(f"drift.warn.{r.status}", name=friendly(r.column), psi=fmt.num(r.psi, 2)),
                            "risk" if r.status == "red" else "warn")
    for r in order.itertuples():
        delta = (r.missing_new - r.missing_ref) * 100
        if abs(delta) >= 10:
            insight_callout(t("drift.warn.missing", name=friendly(r.column), ref=fmt.pct(r.missing_ref * 100, 0),
                              new=fmt.pct(r.missing_new * 100, 0)), "warn")
        if r.unseen:
            insight_callout(t("drift.warn.unseen", name=friendly(r.column), values=", ".join(r.unseen[:5])), "warn")

    # ------------------------------------------------------------ charts
    section_header(t("drift.chart.title"), t("drift.chart.sub"))
    bars = order.sort_values("psi")
    fig = go.Figure(go.Bar(y=[friendly(c) for c in bars["column"]], x=bars["psi"], orientation="h",
                           marker_color=[color[s] for s in bars["status"]], customdata=bars["status"],
                           text=[fmt.num(v, 3) for v in bars["psi"]], textposition="outside",
                           hovertemplate="%{y}: PSI %{x:.3f}<extra></extra>"))
    for x, name in ((drift.GREEN, "0.10"), (drift.AMBER, "0.25")):
        fig.add_vline(x=x, line=dict(color=charts.ink("muted"), width=1), annotation_text=name,
                      annotation_font=dict(size=10, color=charts.ink("secondary")))
    charts.style(fig, height=max(320, 26 * len(bars) + 80), legend=False, xgrid=True, ygrid=False)
    fig.update_xaxes(title="PSI", range=[0, max(float(bars["psi"].max()) * 1.2, 0.3)])
    worst = order.iloc[0]
    charts.show(fig, t("drift.chart.so_what", name=friendly(worst["column"]), psi=fmt.num(worst["psi"], 3),
                       status=t(f"drift.status.{worst['status']}")),
                data=order[["column", "psi", "status", "missing_ref", "missing_new"]].round(4), key="drift_psi")

    pick = st.selectbox(t("drift.chart.pick"), list(order["column"]), format_func=lambda c: f"{friendly(c)} ({c})",
                        key="drift_pick")
    dist = drift.distribution_frame(df, reference, pick, t("cc.form.not_recorded"), t("drift.chart.unseen"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dist["bin"], y=dist["training"] * 100, name=t("drift.chart.training"),
                         marker_color=charts.neutral(), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=dist["bin"], y=dist["batch"] * 100, name=t("drift.chart.batch"),
                         marker_color=charts.accent(), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    charts.style(fig, height=380, pct_y=True)
    fig.update_layout(barmode="group")
    gap = (dist["batch"] - dist["training"]).abs()
    top_bin = dist.loc[gap.idxmax()]
    charts.show(fig, t("drift.chart.dist_so_what", bin=top_bin["bin"], train=fmt.pct(top_bin["training"] * 100, 1),
                       batch=fmt.pct(top_bin["batch"] * 100, 1)),
                data=dist.assign(training=lambda d: (d["training"] * 100).round(2),
                                 batch=lambda d: (d["batch"] * 100).round(2)), key="drift_dist")
    if batch["synthetic"]:
        st.caption(t("drift.synthetic_note"))
