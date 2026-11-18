"""The Problem: why churn matters, who this is for, success criteria, CRISP-DM."""

import html

import streamlit as st

from app import navigation
from app.core import artifacts, fmt, metrics, portfolio
from app.core.segments import with_names
from app.i18n import t
from app.ui.components import insight_callout, kpi_row, page_header, section_header

profiles = with_names(artifacts.get_segment_profiles())
meta = artifacts.get_metadata()
pf = portfolio.compute(profiles, meta)

page_header(t("prob.title"), t("prob.sub"), t("prob.eyebrow"))

# ---------------------------------------------------------------------------
# The business problem
# ---------------------------------------------------------------------------
section_header(t("prob.why.title"), t("prob.why.sub"))
c1, c2 = st.columns([1.3, 1])
with c1:
    for i in range(1, 5):
        st.markdown(f"- {t(f'prob.why.p{i}')}")
with c2:
    kpi_row([
        dict(label=t("kpi.churn_rate"), value=fmt.pct(pf.churn_rate, 2), foot=t("prob.why.kpi_rate_foot")),
        dict(label=t("kpi.est_churners"), value=fmt.num(pf.est_churners), foot=t("prob.why.kpi_churners_foot")),
        dict(label=t("kpi.revenue_concentration"), value=fmt.pct(pf.revenue_concentration, 1),
             foot=t("prob.why.kpi_rev_foot")),
    ], cols=1)

# ---------------------------------------------------------------------------
# Users and decisions
# ---------------------------------------------------------------------------
section_header(t("prob.users.title"), t("prob.users.sub"))
cols = st.columns(3)
for col, role, pages in zip(cols, ("executive", "retention", "analyst"),
                            (["executive", "revenue_risk", "segments"],
                             ["journey", "batch", "retention", "experiment", "campaigns"],
                             ["model", "evaluation", "drift", "methodology"])):
    with col:
        with st.container(border=True, height="stretch"):
            st.html(f'<div class="hub-persona"><div class="who">{html.escape(t(f"home.persona.{role}.who"))}</div>'
                    f'<h4>{html.escape(t(f"prob.users.{role}.question"))}</h4>'
                    f'<p>{html.escape(t(f"prob.users.{role}.decisions"))}</p></div>')
            for key in pages:
                navigation.link(key)

# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------
section_header(t("prob.success.title"), t("prob.success.sub"))
comparison = artifacts.get_optional("model_comparison")
val = artifacts.get_optional("validation_predictions")
rows = []
baseline_auc = None
if comparison is not None:
    lr = comparison[comparison["model"].str.contains("Logistic", case=False)]
    baseline_auc = float(lr["roc_auc"].iloc[0]) if len(lr) else None
auc = float(meta["validation_roc_auc"])
if baseline_auc is not None:
    rows.append((t("prob.success.c1"), t("prob.success.c1_target_lr", lr=fmt.num(baseline_auc, 4)), fmt.num(auc, 4),
                 auc > baseline_auc))
else:
    rows.append((t("prob.success.c1"), t("prob.success.c1_target"), fmt.num(auc, 4), auc > 0.5))
if val is not None:
    cal = metrics.calibration_table(val["y_true"], val["y_prob"], 10)
    gap = float((cal["observed"] - cal["mean_pred"]).abs().max() * 100)
    rows.append((t("prob.success.c2"), t("prob.success.c2_target"), f"{fmt.num(gap, 1)} pp", gap <= 5))
else:
    rows.append((t("prob.success.c2"), t("prob.success.c2_target"), t("prob.success.pending_export"), None))
rows.append((t("prob.success.c3"), t("prob.success.c3_target"), t("prob.success.c3_done"), True))
log = st.session_state.get("campaign_log")
if log is not None and len(log) and "retained" in log and log["retained"].notna().any():
    kept = int(log["retained"].fillna(0).sum())
    rows.append((t("prob.success.c4"), t("prob.success.c4_target"), t("prob.success.c4_logged", n=fmt.num(kept)), kept > 0))
else:
    rows.append((t("prob.success.c4"), t("prob.success.c4_target"), t("prob.success.c4_pending"), None))
status = {True: ("good", "✓ " + t("prob.success.met")), False: ("bad", "✗ " + t("prob.success.not_met")),
          None: ("neutral", "… " + t("prob.success.open"))}
head = "".join(f"<th>{html.escape(t(f'prob.success.col_{c}'))}</th>" for c in ("criterion", "target", "achieved", "status"))
body = "".join(
    f"<tr><td><b>{html.escape(r[0])}</b></td><td>{html.escape(r[1])}</td><td>{html.escape(r[2])}</td>"
    f'<td><span class="hub-kpi-delta {status[r[3]][0]}">{html.escape(status[r[3]][1])}</span></td></tr>'
    for r in rows)
st.html(f'<table class="hub-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')
st.caption(t("prob.success.caption"))

# ---------------------------------------------------------------------------
# CRISP-DM
# ---------------------------------------------------------------------------
section_header(t("prob.crisp.title"), t("prob.crisp.sub"))
STEPS = [("business", "problem"), ("data", "data_explorer"), ("prep", "methodology"), ("model", "model"),
         ("eval", "evaluation"), ("deploy", "batch")]
cards = "".join(
    f'<div class="hub-step"><div class="n">{i:02d}</div><div class="t">{html.escape(t(f"prob.crisp.{k}.name"))}</div>'
    f'<div class="d">{html.escape(t(f"prob.crisp.{k}.desc", rows=fmt.compact(meta["training_rows"]), n=meta["feature_count"], cols=len(artifacts.get_config()["raw_source_columns"])))}</div></div>'
    for i, (k, _) in enumerate(STEPS, 1))
st.html(f'<div class="hub-steps">{cards}</div>')
link_cols = st.columns(6)
for col, (k, page) in zip(link_cols, STEPS):
    with col:
        navigation.link(page, label=t("prob.crisp.go"), help=t(f"prob.crisp.{k}.name"))
insight_callout(t("prob.crisp.loop"), "info")
