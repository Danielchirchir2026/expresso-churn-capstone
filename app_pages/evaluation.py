"""Model Evaluation: ROC, PR, calibration, thresholds and gains on labelled data."""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, evaluation, fmt, metrics, state
from app.i18n import t
from app.ui import charts
from app.ui.components import empty_state, insight_callout, kpi_row, page_header, section_header
from app.ui.guards import require_artifact

bundle = artifacts.get_bundle()
meta = artifacts.get_metadata()
page_header(t("eval.title"), t("eval.sub"), t("eval.eyebrow"))

# ---------------------------------------------------------------------------
# Model comparison (model_comparison.csv)
# ---------------------------------------------------------------------------
section_header(t("eval.cmp.title"), t("eval.cmp.sub"))
comparison = require_artifact("model_comparison", "eval.cmp.missing")
if comparison is not None:
    comp = comparison.copy()
    deployed = comp[comp["model"].str.contains("deployed", case=False)]
    honest = comp[comp["model"].str.contains("holdout", case=False)]
    lgb = honest.iloc[0] if len(honest) else (deployed.iloc[0] if len(deployed) else None)
    base = comp[comp["model"].str.contains("Logistic", case=False)]
    order = comp.sort_values("roc_auc")
    fig = go.Figure(go.Bar(
        y=order["model"], x=order["roc_auc"], orientation="h",
        marker_color=[charts.accent() if "LightGBM" in m else charts.neutral() for m in order["model"]],
        text=[fmt.num(v, 4) for v in order["roc_auc"]], textposition="outside",
        customdata=order[["pr_auc", "log_loss", "f1_at_0_5"]],
        hovertemplate="<b>%{y}</b><br>ROC-AUC %{x:.4f}<br>PR-AUC %{customdata[0]:.4f}<br>"
                      "log loss %{customdata[1]:.4f}<br>F1@0.5 %{customdata[2]:.3f}<extra></extra>"))
    charts.style(fig, height=110 + 48 * len(order), legend=False, xgrid=True, ygrid=False)
    fig.update_xaxes(range=[0.45, 1.06], title="ROC-AUC")
    so = t("eval.cmp.so_what_nobase")
    if lgb is not None and len(base):
        b = base.iloc[0]
        so = t("eval.cmp.so_what", auc=fmt.num(lgb["roc_auc"], 4), lr=fmt.num(b["roc_auc"], 4),
               ap=fmt.num(lgb["pr_auc"], 3), lrap=fmt.num(b["pr_auc"], 3), gain=fmt.num(lgb["roc_auc"] - b["roc_auc"], 3))
    c1, c2 = st.columns([1.2, 1])
    with c1:
        charts.show(fig, so, key="ev_cmp")
    with c2:
        st.dataframe(comp, hide_index=True, width="stretch", column_config={
            "model": st.column_config.TextColumn(t("eval.cmp.col_model"), width="medium"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
            "log_loss": st.column_config.NumberColumn(t("eval.cmp.col_logloss"), format="%.4f"),
            "f1_at_0_5": st.column_config.NumberColumn("F1 @ 0.5", format="%.3f"),
            "train_seconds": st.column_config.NumberColumn(t("eval.cmp.col_time"), format="%.1f s"),
            "note": st.column_config.TextColumn(t("eval.cmp.col_note"), width="large"),
        })
    insight_callout(t("eval.cmp.why"), "info", tag=t("eval.cmp.why_tag"))
    if len(honest) and len(deployed):
        insight_callout(t("eval.cmp.honest", dep=fmt.num(deployed.iloc[0]["roc_auc"], 4),
                          hon=fmt.num(honest.iloc[0]["roc_auc"], 4)), "warn")

# ---------------------------------------------------------------------------
# Choose labelled data
# ---------------------------------------------------------------------------
section_header(t("eval.data.title"), t("eval.data.sub"))
val_df = artifacts.get_optional("validation_predictions")
batch = state.get_batch()
batch_has_label = bool(batch and "CHURN" in batch["df"].columns)
options = (["validation"] if val_df is not None else []) + ["upload"] + (["batch"] if batch_has_label else [])
st.session_state.setdefault("eval_source", options[0])
if st.session_state["eval_source"] not in options:
    st.session_state["eval_source"] = options[0]
source = st.segmented_control(t("eval.source.label"), options, key="eval_source",
                              format_func=lambda o: t(f"eval.source.{o}")) or options[0]

if val_df is None:
    empty_state(t("ui.missing_artifact", file="artifacts/validation_predictions.parquet"),
                t("eval.source.no_artifact"), command=artifacts.EXPORT_COMMAND, icon="▤")


@st.cache_data(show_spinner=False, max_entries=2)
def read_any(data: bytes, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data)) if name.lower().endswith(".parquet") else pd.read_csv(io.BytesIO(data))


data = None
if source == "validation" and val_df is not None:
    data = evaluation.from_validation(val_df)
elif source == "batch" and batch_has_label:
    data = evaluation.from_frame(batch["df"], bundle, batch["name"], source="batch")
elif source == "upload":
    up = st.file_uploader(t("eval.upload.label"), type=["csv", "parquet"], help=t("eval.upload.help"), key="eval_up")
    if up is not None:
        cache = st.session_state.get("eval_upload")
        if cache and cache[0] == up.file_id:
            data = cache[1]
        else:
            try:
                bar = st.progress(0.0, text=t("batch.score.progress"))
                data = evaluation.from_frame(read_any(up.getvalue(), up.name), bundle, up.name,
                                             progress=lambda f: bar.progress(f, text=t("batch.score.progress")))
                bar.empty()
                st.session_state["eval_upload"] = (up.file_id, data)
            except evaluation.EvalDataError as err:
                st.error(t(f"eval.err.{err.code}", **err.params), icon=":material/error:")
    else:
        st.caption(t("eval.upload.hint"))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if data is not None:
    s = evaluation.summary(data)
    section_header(t("eval.summary.title", name=data.name), t(f"eval.summary.sub_{data.source}"))
    kpi_row([
        dict(label=t("eval.kpi.rows"), value=fmt.num(s["n"])),
        dict(label=t("eval.kpi.rate"), value=fmt.pct(s["rate"] * 100, 2)),
        dict(label="ROC-AUC", value=fmt.num(s["auc"], 4), help=t("glossary.roc_auc.def"),
             foot=t("eval.kpi.meta_auc", auc=fmt.num(meta["validation_roc_auc"], 4))),
        dict(label=t("eval.kpi.ap"), value=fmt.num(s["ap"], 4), help=t("eval.kpi.ap_help"),
             foot=t("eval.kpi.ap_base", base=fmt.num(s["rate"], 3))),
        dict(label=t("eval.kpi.brier"), value=fmt.num(s["brier"], 4), help=t("eval.kpi.brier_help")),
    ], cols=5)
    if data.source == "validation":
        manifest = artifacts.get_optional("export_manifest") or {}
        if manifest.get("validation_model") == "deployed" and s["auc"] > meta["validation_roc_auc"] + 0.002:
            insight_callout(t("eval.summary.optimistic", auc=fmt.num(s["auc"], 4),
                              meta=fmt.num(meta["validation_roc_auc"], 4)), "warn")
        elif manifest.get("validation_model") == "holdout_refit":
            insight_callout(t("eval.summary.holdout"), "info")

    tab_curves, tab_thr, tab_gains, tab_fair = st.tabs([t("eval.tab.curves"), t("eval.tab.threshold"),
                                                        t("eval.tab.gains"), t("eval.tab.fairness")])

    # ---------------- Curves -------------------------------------------------
    with tab_curves:
        c1, c2 = st.columns(2)
        with c1:
            roc = evaluation.roc_points(data)
            f = go.Figure()
            f.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"ROC (AUC {s['auc']:.4f})",
                                   line=dict(color=charts.accent(), width=2), customdata=roc["threshold"],
                                   hovertemplate="FPR %{x:.3f} · TPR %{y:.3f}<br>t = %{customdata:.3f}<extra></extra>"))
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name=t("ret.gains.random"),
                                   line=dict(color=charts.ink("muted"), width=1)))
            charts.style(f, height=380, xgrid=True, title=t("eval.roc.title"))
            f.update_xaxes(title=t("eval.roc.x"), range=[0, 1])
            f.update_yaxes(title=t("eval.roc.y"), range=[0, 1.02])
            charts.show(f, t("eval.roc.so_what", auc=fmt.num(s["auc"], 3), pct=fmt.pct(s["auc"] * 100, 1)),
                        key="ev_roc")
        with c2:
            pr = evaluation.pr_points(data)
            f = go.Figure()
            f.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=f"AP {s['ap']:.4f}",
                                   line=dict(color=charts.accent(), width=2), customdata=pr["threshold"],
                                   hovertemplate=t("eval.pr.hover") + "<extra></extra>"))
            f.add_hline(y=s["rate"], line=dict(color=charts.ink("muted"), width=1),
                        annotation_text=t("eval.pr.baseline", rate=fmt.pct(s["rate"] * 100, 1)),
                        annotation_font=dict(size=11, color=charts.ink("secondary")))
            charts.style(f, height=380, xgrid=True, title=t("eval.pr.title"))
            f.update_xaxes(title=t("eval.pr.x"), range=[0, 1])
            f.update_yaxes(title=t("eval.pr.y"), range=[0, 1.02])
            at80 = pr[pr["recall"] >= 0.8]["precision"].max() if (pr["recall"] >= 0.8).any() else float("nan")
            charts.show(f, t("eval.pr.so_what", p=fmt.pct(at80 * 100, 0), base=fmt.pct(s["rate"] * 100, 1)),
                        key="ev_pr")
        cal = metrics.calibration_table(data.y, data.p, 10)
        f = go.Figure()
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name=t("eval.cal.perfect"),
                               line=dict(color=charts.ink("muted"), width=1)))
        f.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["observed"], mode="lines+markers", name=t("eval.cal.model"),
                               line=dict(color=charts.accent(), width=2), marker=dict(size=9),
                               customdata=cal["n"],
                               hovertemplate=t("eval.cal.hover") + "<extra></extra>"))
        charts.style(f, height=380, xgrid=True, title=t("eval.cal.title"))
        f.update_xaxes(title=t("eval.cal.x"), range=[0, 1])
        f.update_yaxes(title=t("eval.cal.y"), range=[0, 1])
        gap = float((cal["observed"] - cal["mean_pred"]).abs().max())
        charts.show(f, t("eval.cal.so_what", gap=fmt.pp(gap * 100, 1), brier=fmt.num(s["brier"], 4)),
                    data=cal.round(4).rename(columns={"mean_pred": t("eval.cal.x"), "observed": t("eval.cal.y"),
                                                      "n": t("eval.kpi.rows")}), key="ev_cal")

    # ---------------- Threshold ----------------------------------------------
    with tab_thr:
        sweep = metrics.threshold_sweep(data.y, data.p)
        best_f1_t = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])
        thr = st.slider(t("eval.thr.label"), 0.0, 1.0, 0.5, 0.005, key="eval_thr", format="%.3f",
                        help=t("eval.thr.help"))
        st.caption(t("eval.thr.f1_best", t=fmt.num(best_f1_t, 3)) + " · " + t("ui.exploratory_note"))
        cm = metrics.confusion_at(data.y, data.p, thr)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            z = [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]]
            text = [[t("eval.cm.tp", n=fmt.num(cm["tp"])), t("eval.cm.fn", n=fmt.num(cm["fn"]))],
                    [t("eval.cm.fp", n=fmt.num(cm["fp"])), t("eval.cm.tn", n=fmt.num(cm["tn"]))]]
            hm = go.Figure(go.Heatmap(z=z, x=[t("eval.cm.pred_churn"), t("eval.cm.pred_stay")],
                                      y=[t("eval.cm.actual_churn"), t("eval.cm.actual_stay")], text=text,
                                      texttemplate="%{text}", colorscale=charts.sequential(), showscale=False,
                                      hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>", xgap=3, ygap=3))
            charts.style(hm, height=330, legend=False, ygrid=False, title=t("eval.cm.title", t=fmt.num(thr, 3)))
            hm.update_yaxes(autorange="reversed")
            hm.update_traces(textfont=dict(size=13))
            charts.show(hm, t("eval.cm.so_what", flagged=fmt.num(cm["flagged"]), tp=fmt.num(cm["tp"]),
                              fp=fmt.num(cm["fp"])), key="ev_cm")
        with c2:
            kpi_row([
                dict(label=t("eval.thr.precision"), value=fmt.pct(cm["precision"] * 100, 1), help=t("eval.thr.precision_help")),
                dict(label=t("eval.thr.recall"), value=fmt.pct(cm["recall"] * 100, 1), help=t("eval.thr.recall_help")),
                dict(label="F1", value=fmt.num(cm["f1"], 3)),
                dict(label=t("eval.thr.flagged"), value=fmt.num(cm["flagged"]),
                     foot=t("eval.thr.flagged_share", share=fmt.pct(cm["flagged"] / max(s["n"], 1) * 100, 1))),
            ], cols=2)
            if not np.isnan(cm["precision"]):
                insight_callout(t("eval.thr.words", p=fmt.num(cm["precision"] * 100, 0),
                                  r=fmt.num(cm["recall"] * 100, 0), fp=fmt.num(cm["fp"]), fn=fmt.num(cm["fn"])),
                                "info", tag=t("eval.thr.words_tag"))
        f = go.Figure()
        for col, name, color in (("precision", t("eval.thr.precision"), charts.series(0)),
                                 ("recall", t("eval.thr.recall"), charts.series(1)),
                                 ("f1", "F1", charts.accent())):
            f.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[col], mode="lines", name=name,
                                   line=dict(color=color, width=2),
                                   hovertemplate=name + " %{y:.3f} at t = %{x:.3f}<extra></extra>"))
        f.add_vline(x=thr, line=dict(color=charts.ink("muted"), width=1))
        charts.style(f, height=340, xgrid=True, title=t("eval.thr.curve_title"))
        f.update_xaxes(title=t("eval.thr.label"))
        f.update_yaxes(range=[0, 1.02])
        charts.show(f, t("eval.thr.curve_so_what", t=fmt.num(best_f1_t, 3),
                         f1=fmt.num(sweep["f1"].max(), 3)), key="ev_sweep")

        # ------------- Threshold advisor (cost-based) -------------------------
        section_header(t("eval.adv.title"), t("eval.adv.sub"))
        profiles = artifacts.get_segment_profiles()
        typical = float((profiles["customers"] * profiles["median_revenue"]).sum() / profiles["customers"].sum())
        a1, a2 = st.columns(2)
        with a1:
            c_fp = st.number_input(t("eval.adv.cost_fp"), 0.0, 1e7, float(st.session_state.get("ret_cost", 500.0)),
                                   50.0, key="adv_fp", help=t("eval.adv.cost_fp_help"))
        with a2:
            c_fn = st.number_input(t("eval.adv.cost_fn"), 0.0, 1e8, float(round(typical * 3)), 100.0, key="adv_fn",
                                   help=t("eval.adv.cost_fn_help", typical=fmt.num(typical)))
        costs = metrics.threshold_cost(data.y, data.p, c_fp, c_fn)
        best_t = metrics.best_cost_threshold(costs)
        best = costs.loc[costs["threshold"] == best_t].iloc[0]
        nobody = float(data.y.sum()) * c_fn
        insight_callout(t("eval.adv.result", t=fmt.num(best_t, 3), flagged=fmt.num(best["flagged"]),
                          cost=fmt.money(best["cost"]), nobody=fmt.money(nobody),
                          saving=fmt.money(nobody - best["cost"])), "good", tag=t("eval.adv.tag"))
        f = go.Figure()
        f.add_trace(go.Scatter(x=costs["threshold"], y=costs["cost"], mode="lines", name=t("eval.adv.total"),
                               line=dict(color=charts.accent(), width=2),
                               hovertemplate="t = %{x:.3f}<br>%{y:,.0f} FCFA<extra></extra>"))
        f.add_trace(go.Scatter(x=[best_t], y=[best["cost"]], mode="markers", name=t("eval.adv.best"),
                               marker=dict(size=11, color=charts.accent(),
                                           line=dict(width=2, color=charts.ink("surface"))),
                               hovertemplate="t = %{x:.3f}<br>%{y:,.0f} FCFA<extra></extra>"))
        charts.style(f, height=340, xgrid=True, title=t("eval.adv.chart"))
        f.update_xaxes(title=t("eval.thr.label"))
        f.update_yaxes(title="FCFA")
        charts.show(f, t("eval.adv.so_what", ratio=fmt.num(c_fn / c_fp, 1) if c_fp else "-", t=fmt.num(best_t, 3)),
                    key="ev_adv")
        st.caption(t("eval.adv.caption", n=fmt.num(s["n"])))

    # ---------------- Gains & lift -------------------------------------------
    with tab_gains:
        x, y = metrics.gains_curve(data.p, data.y)
        idx = np.unique(np.linspace(0, len(x) - 1, min(len(x), 500)).round().astype(int))
        c20 = metrics.captured_at(data.p, data.y, 0.2) * 100
        f = go.Figure()
        f.add_trace(go.Scatter(x=x[idx] * 100, y=y[idx] * 100, mode="lines", name=t("ret.gains.model"),
                               line=dict(color=charts.accent(), width=2),
                               hovertemplate="%{x:.1f}% → %{y:.1f}%<extra></extra>"))
        f.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name=t("ret.gains.random"),
                               line=dict(color=charts.ink("muted"), width=1)))
        f.add_trace(go.Scatter(x=[20], y=[c20], mode="markers+text", text=[fmt.pct(c20, 0)], textposition="top left",
                               marker=dict(size=10, color=charts.accent()), showlegend=False,
                               hovertemplate="20% → %{y:.1f}%<extra></extra>"))
        charts.style(f, height=380, pct_x=True, pct_y=True, xgrid=True, title=t("eval.gains.title"))
        f.update_xaxes(title=t("ret.gains.x"))
        f.update_yaxes(title=t("eval.gains.y"))
        charts.show(f, t("eval.gains.so_what", cap=fmt.pct(c20, 0), lift=fmt.num(c20 / 20, 1)), key="ev_gains")
        gt = metrics.gains_table(data.p, data.y)
        st.dataframe(gt, hide_index=True, width="stretch", column_config={
            "bin": st.column_config.NumberColumn(t("batchui.c360.decile")),
            "customers": st.column_config.NumberColumn(t("eval.kpi.rows"), format="localized"),
            "outcomes": st.column_config.NumberColumn(t("eval.gains.churners"), format="localized"),
            "rate": st.column_config.NumberColumn(t("eval.gains.rate"), format="percent"),
            "cum_captured": st.column_config.NumberColumn(t("eval.gains.cum"), format="percent"),
            "lift": st.column_config.NumberColumn(t("eval.gains.lift"), format="%.2f×"),
            "cum_lift": st.column_config.NumberColumn(t("eval.gains.cum_lift"), format="%.2f×"),
        })

    # ---------------- Fairness ------------------------------------------------
    with tab_fair:
        if not data.groups:
            st.info(t("eval.fair.no_groups"), icon=":material/info:")
        else:
            st.caption(t("eval.fair.intro", t=fmt.num(thr, 3)))
            for gname, values in data.groups.items():
                gm = metrics.group_metrics(data.y, data.p, values, thr)
                section_header(t("eval.fair.by", name=t(f"feat.{gname}.name")))
                flagged = gm[gm["flagged"]]
                if len(flagged):
                    insight_callout(t("eval.fair.flagged", groups=", ".join(flagged["group"]),
                                      overall=fmt.num(s["auc"], 3)), "risk")
                else:
                    insight_callout(t("eval.fair.ok", overall=fmt.num(s["auc"], 3)), "good")
                show = gm[~gm["too_small"]].sort_values("auc")
                f = go.Figure(go.Bar(
                    y=show["group"], x=show["auc"], orientation="h",
                    marker_color=[charts.risk_up() if fl else charts.accent() for fl in show["flagged"]],
                    customdata=show[["n", "recall", "mean_pred", "churn_rate"]],
                    hovertemplate="<b>%{y}</b><br>AUC %{x:.3f}<br>n = %{customdata[0]:,}<br>"
                                  + t("eval.thr.recall") + " %{customdata[1]:.2f}<br>"
                                  + t("eval.fair.mean_pred") + " %{customdata[2]:.3f}<br>"
                                  + t("eval.kpi.rate") + " %{customdata[3]:.3f}<extra></extra>"))
                f.add_vline(x=s["auc"], line=dict(color=charts.ink("muted"), width=1),
                            annotation_text=t("eval.fair.overall", auc=fmt.num(s["auc"], 3)),
                            annotation_font=dict(size=11, color=charts.ink("secondary")))
                f.add_vline(x=s["auc"] - 0.05, line=dict(color=charts.risk_up(), width=1))
                charts.style(f, height=max(260, 26 * len(show) + 80), legend=False, xgrid=True, ygrid=False)
                lo = max(0.5, float(show["auc"].min()) - 0.03) if len(show) else 0.5
                f.update_xaxes(range=[lo, 1.0], title="ROC-AUC")
                if len(show):
                    worst = show.iloc[0]
                    so = t("eval.fair.so_what", worst=worst["group"], auc=fmt.num(worst["auc"], 3),
                           overall=fmt.num(s["auc"], 3))
                else:
                    so = t("eval.fair.no_groups")
                charts.show(f, so, data=gm.round(4), key=f"ev_fair_{gname}")
            st.caption(t("eval.fair.caption"))
