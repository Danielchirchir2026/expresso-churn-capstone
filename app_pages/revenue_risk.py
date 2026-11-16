"""Revenue at Risk: portfolio estimate by segment, and expected value at risk in a batch."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, fmt, state
from app.core.features import friendly
from app.core.segments import SEGMENT_IDS, segment_name, with_names
from app.i18n import t
from app.ui import charts
from app.ui.batchui import require_batch
from app.ui.components import insight_callout, kpi_row, page_header, section_header

profiles = with_names(artifacts.get_segment_profiles())
page_header(t("rar.title"), t("rar.sub"), t("rar.eyebrow"))

# ---------------------------------------------------------------------------
# Portfolio: customers x churn rate x median monthly revenue, per segment
# ---------------------------------------------------------------------------
section_header(t("rar.port.title"), t("rar.port.sub"))
p = profiles.set_index("segment")
risk = (p["customers"] * p["churn_rate"] / 100 * p["median_revenue"]).reindex(SEGMENT_IDS)
total = float(risk.sum())
names = [segment_name(s) for s in SEGMENT_IDS]
wf = go.Figure(go.Waterfall(
    x=names + [t("rar.port.total")], y=list(risk.values) + [0], measure=["relative"] * len(SEGMENT_IDS) + ["total"],
    text=[fmt.compact(v) for v in risk.values] + [fmt.compact(total)], textposition="outside",
    increasing=dict(marker=dict(color=charts.risk_up())), totals=dict(marker=dict(color=charts.accent())),
    connector=dict(line=dict(color=charts.ink("axis"), width=1)),
    customdata=np.array([[int(p.loc[s, "customers"]), p.loc[s, "churn_rate"], p.loc[s, "median_revenue"]]
                         for s in SEGMENT_IDS] + [[np.nan, np.nan, np.nan]]),
    hovertemplate="<b>%{x}</b><br>%{y:,.0f} FCFA<br>" + t("rar.port.hover") + "<extra></extra>"))
charts.style(wf, height=430, legend=False)
wf.update_yaxes(title=t("rar.port.axis"))
top = int(risk.idxmax())
charts.show(wf, t("rar.port.so_what", total=fmt.money(total), seg=segment_name(top),
                  share=fmt.pct(risk.max() / total * 100 if total else 0, 0)),
            data=pd.DataFrame({t("exec.col.segment"): names, t("exec.hover.customers"): p.loc[SEGMENT_IDS, "customers"].values,
                               t("exec.axis.churn_rate"): p.loc[SEGMENT_IDS, "churn_rate"].round(2).values,
                               t("rar.port.median"): p.loc[SEGMENT_IDS, "median_revenue"].values,
                               t("rar.port.axis"): risk.round(0).values}), key="rar_wf")
for s in SEGMENT_IDS:
    if p.loc[s, "median_revenue"] == 0:
        insight_callout(t("rar.port.zero", seg=segment_name(s), churn=fmt.pct(p.loc[s, "churn_rate"], 1)), "warn",
                        tag=t("rar.port.zero_tag"))
st.caption(t("rar.port.caption"))

# ---------------------------------------------------------------------------
# Batch: expected revenue at risk = sum of probability x REVENUE
# ---------------------------------------------------------------------------
section_header(t("rar.batch.title"), t("rar.batch.sub"))
batch = require_batch("rar")
if batch:
    df = batch["df"]
    rev = pd.to_numeric(df["REVENUE"], errors="coerce").fillna(0)
    ear = df["churn_probability"] * rev
    total_rev = float(rev.sum())
    kpi_row([
        dict(label=t("rar.batch.total"), value=fmt.money(ear.sum()), help=t("rar.batch.total_help")),
        dict(label=t("rar.batch.share"), value=fmt.pct(ear.sum() / total_rev * 100 if total_rev else 0, 1),
             foot=t("rar.batch.of_revenue", rev=fmt.money(total_rev))),
        dict(label=t("rar.batch.top10"), value=fmt.pct(ear.nlargest(max(len(df) // 10, 1)).sum() / ear.sum() * 100
                                                       if ear.sum() else 0, 0), help=t("rar.batch.top10_help")),
        dict(label=t("rar.batch.no_rev"), value=fmt.pct((rev <= 0).mean() * 100, 1), help=t("rar.batch.no_rev_help")),
    ], cols=4)
    region = df["REGION"].fillna(t("cc.form.not_recorded")).astype(str)
    c1, c2 = st.columns(2)
    for col, key, groups, order in ((c1, "region", region, None), (c2, "tenure", df["TENURE"].astype(str),
                                                                   artifacts.get_config()["tenure_order"])):
        with col:
            g = pd.DataFrame({"k": groups.to_numpy(), "ear": ear.to_numpy()}).groupby("k")["ear"].agg(["sum", "size"])
            g = g.reindex([o for o in order if o in g.index]) if order else g.sort_values("sum")
            horizontal = order is None
            fig = go.Figure(go.Bar(x=g["sum"] if horizontal else g.index, y=g.index if horizontal else g["sum"],
                                   orientation="h" if horizontal else "v", marker_color=charts.risk_up(),
                                   customdata=g["size"],
                                   hovertemplate="%{" + ("y" if horizontal else "x") + "}: %{" + ("x" if horizontal else "y")
                                                 + ":,.0f} FCFA<br>n = %{customdata:,}<extra></extra>"))
            charts.style(fig, height=380, legend=False, xgrid=horizontal, ygrid=not horizontal,
                         title=t(f"rar.batch.by_{key}"))
            top_k = g["sum"].idxmax()
            charts.show(fig, t("rar.batch.so_what", level=top_k, value=fmt.money(g["sum"].max()),
                               share=fmt.pct(g["sum"].max() / ear.sum() * 100 if ear.sum() else 0, 0)),
                        data=g.reset_index().rename(columns={"k": friendly("REGION" if key == "region" else "TENURE"),
                                                             "sum": t("rar.batch.total"), "size": t("batch.kpi.customers")}
                                                    ).round(0), key=f"rar_{key}")
    seg = pd.DataFrame({"s": df["segment"].to_numpy(), "ear": ear.to_numpy()}).groupby("s")["ear"].sum()
    st.caption(t("rar.batch.by_segment", parts=" · ".join(f"{segment_name(s)} {fmt.money(v)}" for s, v in seg.items())))
    st.caption(t("rar.batch.caption"))
