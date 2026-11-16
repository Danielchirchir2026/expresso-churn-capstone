"""Retention Planner: how many customers to contact for the best expected return."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, fmt, metrics
from app.core.features import friendly
from app.core.segments import segment_name
from app.i18n import t
from app.ui import charts
from app.ui.batchui import require_batch
from app.ui.components import insight_callout, kpi_row, page_header, section_header

config = artifacts.get_config()
ID = config["id_column"]

page_header(t("ret.title"), t("ret.sub"), t("ret.eyebrow"))
batch = require_batch("ret")

if batch:
    df = batch["df"]
    n = len(df)
    p = df["churn_probability"].to_numpy()
    rev = pd.to_numeric(df["REVENUE"], errors="coerce").to_numpy()
    median_rev = float(np.nanmedian(rev)) if np.isfinite(rev).any() else float("nan")

    # -----------------------------------------------------------------------
    # Assumptions
    # -----------------------------------------------------------------------
    section_header(t("ret.inputs.title"), t("ret.inputs.sub"))
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.radio(t("ret.inputs.value_mode"), ["per_customer", "flat"],
                            format_func=lambda m: t(f"ret.inputs.mode.{m}"), key="ret_mode",
                            help=t("ret.inputs.value_mode_help"))
            flat = st.number_input(t("ret.inputs.flat"), 0.0, 1e7,
                                   float(round(median_rev)) if np.isfinite(median_rev) else 0.0, 100.0, key="ret_flat",
                                   disabled=mode != "flat",
                                   help=t("ret.inputs.flat_help", median=fmt.num(median_rev)))
        with c2:
            cost = st.number_input(t("ret.inputs.cost"), 0.0, 1e6, 500.0, 50.0, key="ret_cost",
                                   help=t("ret.inputs.cost_help"))
            save = st.slider(t("ret.inputs.save"), 0.0, 1.0, 0.20, 0.01, key="ret_save", format="%.2f",
                             help=t("ret.inputs.save_help"))
        with c3:
            capacity = st.number_input(t("ret.inputs.capacity"), 1, max(n, 1), min(n, 5000), 100, key="ret_capacity",
                                       help=t("ret.inputs.capacity_help"))
            horizon = st.slider(t("ret.inputs.horizon"), 1, 24, 3, key="ret_horizon", help=t("ret.inputs.horizon_help"))
        st.session_state.setdefault("ret_rank", "prob")
        rank_by = st.segmented_control(t("ret.inputs.rank"), ["prob", "value"], key="ret_rank",
                                       format_func=lambda r: t(f"ret.inputs.rank_{r}"), help=t("ret.inputs.rank_help"))
        rank_by = rank_by or "prob"

    value = (np.nan_to_num(rev, nan=0.0) if mode == "per_customer" else np.full(n, flat)) * horizon
    plan, order = metrics.retention_plan(p, value, cost, save, rank_by=rank_by)
    k_best = metrics.best_k(plan, int(capacity))

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    section_header(t("ret.rec.title"), t("ret.rec.sub"))
    if k_best == 0:
        insight_callout(t("ret.rec.none"), "warn")
        row = plan.iloc[min(int(capacity), n) - 1]
        other = "value" if rank_by == "prob" else "prob"
        alt_plan, _ = metrics.retention_plan(p, value, cost, save, rank_by=other)
        alt_k = metrics.best_k(alt_plan, int(capacity))
        if alt_k:
            alt = alt_plan.iloc[alt_k - 1]
            zero_rev = float((np.nan_to_num(rev, nan=0.0)[order[: min(int(capacity), n)]] <= 0).mean() * 100)
            insight_callout(t(f"ret.rec.alt_{other}", k=fmt.num(alt_k), net=fmt.money(alt["net"]),
                              zero=fmt.pct(zero_rev, 0)), "info", tag=t("ret.rec.alt_tag"))
            st.button(t(f"ret.rec.switch_{other}"), key="ret_switch", icon=":material/swap_vert:",
                      on_click=lambda o=other: st.session_state.update(ret_rank=o))
    else:
        row = plan.iloc[k_best - 1]
        insight_callout(t("ret.rec.best", k=fmt.num(k_best), net=fmt.money(row["net"]),
                          saved=fmt.num(row["saved"], 0), cap=fmt.num(capacity)), "good")
    roi = row["net"] / row["cost"] * 100 if row["cost"] > 0 else float("nan")
    kpi_row([
        dict(label=t("ret.kpi.contact"), value=fmt.num(k_best)),
        dict(label=t("ret.kpi.reached"), value=fmt.num(row["reached"], 0), help=t("ret.kpi.reached_help")),
        dict(label=t("ret.kpi.saved"), value=fmt.num(row["saved"], 0), help=t("ret.kpi.saved_help")),
        dict(label=t("ret.kpi.revenue"), value=fmt.money(row["revenue_saved"])),
        dict(label=t("ret.kpi.cost"), value=fmt.money(row["cost"])),
        dict(label=t("ret.kpi.net"), value=fmt.money(row["net"]), tone="good" if row["net"] > 0 else "bad",
             delta=t("ret.kpi.roi", roi=fmt.pct(roi, 0)) if np.isfinite(roi) else None),
    ], cols=3)

    # -----------------------------------------------------------------------
    # Profit curve
    # -----------------------------------------------------------------------
    section_header(t("ret.curve.title"), t("ret.curve.sub"))
    ds = metrics.downsample(plan, 700)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds["k"], y=ds["revenue_saved"], name=t("ret.kpi.revenue"), mode="lines",
                             line=dict(color=charts.series(2), width=2),
                             hovertemplate="k = %{x:,}<br>%{y:,.0f} FCFA<extra></extra>"))
    fig.add_trace(go.Scatter(x=ds["k"], y=ds["cost"], name=t("ret.kpi.cost"), mode="lines",
                             line=dict(color=charts.risk_up(), width=2),
                             hovertemplate="k = %{x:,}<br>%{y:,.0f} FCFA<extra></extra>"))
    fig.add_trace(go.Scatter(x=ds["k"], y=ds["net"], name=t("ret.kpi.net"), mode="lines",
                             line=dict(color=charts.accent(), width=3),
                             hovertemplate="k = %{x:,}<br>%{y:,.0f} FCFA<extra></extra>"))
    if k_best:
        fig.add_trace(go.Scatter(x=[k_best], y=[row["net"]], mode="markers", name=t("ret.curve.best"),
                                 marker=dict(size=12, color=charts.accent(), line=dict(width=2, color=charts.ink("surface"))),
                                 hovertemplate=t("ret.curve.best") + ": k = %{x:,}<br>%{y:,.0f} FCFA<extra></extra>"))
    fig.add_vline(x=int(capacity), line=dict(color=charts.ink("muted"), width=1),
                  annotation_text=t("ret.curve.capacity"), annotation_font=dict(size=11, color=charts.ink("secondary")))
    charts.style(fig, height=420, xgrid=True)
    fig.update_xaxes(title=t("ret.curve.x"))
    fig.update_yaxes(title="FCFA")
    peak_all = int(plan.loc[plan["net"].idxmax(), "k"])
    so = (t("ret.curve.so_what", k=fmt.num(k_best), peak=fmt.num(peak_all)) if k_best
          else t("ret.curve.so_what_none"))
    charts.show(fig, so, data=ds.round(1).rename(columns={"k": t("ret.curve.x")}), key="ret_curve")

    # -----------------------------------------------------------------------
    # Gains / lift (expected, from predicted probabilities)
    # -----------------------------------------------------------------------
    section_header(t("ret.gains.title"), t("ret.gains.sub"))
    x, y = metrics.gains_curve(p, p)
    idx = np.unique(np.linspace(0, len(x) - 1, min(len(x), 400)).round().astype(int))
    g = go.Figure()
    g.add_trace(go.Scatter(x=x[idx] * 100, y=y[idx] * 100, mode="lines", name=t("ret.gains.model"),
                           line=dict(color=charts.accent(), width=2),
                           hovertemplate="%{x:.1f}% → %{y:.1f}%<extra></extra>"))
    g.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name=t("ret.gains.random"),
                           line=dict(color=charts.ink("muted"), width=1)))
    for frac in (0.1, 0.2, 0.3):
        cap = metrics.captured_at(p, p, frac) * 100
        g.add_trace(go.Scatter(x=[frac * 100], y=[cap], mode="markers+text", showlegend=False,
                               text=[fmt.pct(cap, 0)], textposition="top left",
                               marker=dict(size=9, color=charts.accent()),
                               hovertemplate=f"{fmt.pct(frac * 100, 0)} → %{{y:.1f}}%<extra></extra>"))
    charts.style(g, height=380, pct_x=True, pct_y=True, xgrid=True)
    g.update_xaxes(title=t("ret.gains.x"))
    g.update_yaxes(title=t("ret.gains.y"))
    c20 = metrics.captured_at(p, p, 0.2) * 100
    gt = metrics.gains_table(p, p)
    charts.show(g, t("ret.gains.so_what", cap=fmt.pct(c20, 0), lift=fmt.num(c20 / 20, 1)),
                data=gt.assign(rate=lambda d: (d["rate"] * 100).round(2), cum_captured=lambda d: (d["cum_captured"] * 100)
                               .round(1)).round(2), key="ret_gains")
    st.caption(t("ret.gains.caveat"))

    # -----------------------------------------------------------------------
    # Targeted list
    # -----------------------------------------------------------------------
    section_header(t("ret.list.title"), t("ret.list.sub"))
    k_export = k_best or min(int(capacity), n)
    chosen = df.iloc[order[:k_export]].copy()
    chosen["expected_value_at_risk"] = (chosen["churn_probability"] * value[order[:k_export]]).round(0)
    chosen["segment_name"] = chosen["segment"].map(lambda s: segment_name(s, "en"))
    for c in ("reason_1", "reason_2", "reason_3"):
        if c in chosen.columns:
            chosen[f"{c}_label"] = chosen[c].map(lambda g: friendly(g, "en") if isinstance(g, str) else "")
    cols = [ID, "churn_probability", "risk_decile", "expected_value_at_risk", "segment_name", "REGION", "TENURE",
            "REVENUE"] + [c for c in chosen.columns if c.endswith("_label")]
    shown = chosen[cols].head(200).copy()
    for c in ("REGION", "TENURE") + tuple(c for c in cols if c.endswith("_label")):
        shown[c] = shown[c].fillna("—").replace("", "—")
    st.dataframe(shown, hide_index=True, width="stretch", height=300, column_config={
        ID: st.column_config.TextColumn(t("cc.form.customer_id")),
        "churn_probability": st.column_config.ProgressColumn(t("ui.gauge.label"), format="%.3f", min_value=0,
                                                             max_value=1),
        "risk_decile": st.column_config.NumberColumn(t("batchui.c360.decile")),
        "expected_value_at_risk": st.column_config.NumberColumn(t("ret.list.value_col"), format="localized"),
        "segment_name": st.column_config.TextColumn(t("cc.result.segment")),
        "REGION": st.column_config.TextColumn(friendly("REGION")),
        "TENURE": st.column_config.TextColumn(friendly("TENURE")),
        "REVENUE": st.column_config.NumberColumn(friendly("REVENUE"), format="localized"),
        "reason_1_label": st.column_config.TextColumn(t("ret.list.reason", n=1)),
        "reason_2_label": st.column_config.TextColumn(t("ret.list.reason", n=2)),
        "reason_3_label": st.column_config.TextColumn(t("ret.list.reason", n=3)),
    })
    st.download_button(t("ret.list.download", k=fmt.num(k_export)),
                       data=lambda: chosen[cols].to_csv(index=False).encode("utf-8"),
                       file_name=f"retention_target_list_top{k_export}.csv", mime="text/csv",
                       icon=":material/download:", type="primary", on_click="ignore")
    st.session_state["ret_last_plan"] = {"k": int(k_best), "save_rate": float(save), "offer_cost": float(cost),
                                         "expected_saved": float(row["saved"]), "expected_net": float(row["net"]),
                                         "horizon": int(horizon)}

    # -----------------------------------------------------------------------
    # Assumptions, stated plainly
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(f"**{t('ret.assume.title')}**")
        for i in range(1, 6):
            st.markdown(f"- {t(f'ret.assume.a{i}')}")
