"""UI pieces that depend on a scored batch being in session."""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd
import streamlit as st

from app import navigation
from app.core import artifacts, batchops, explain, fmt, portfolio, scoring, state
from app.core.features import friendly
from app.core.segments import segment_name
from app.i18n import t
from app.ui import charts
from app.ui.components import empty_state, insight_callout, kpi_row, prob_gauge, synthetic_banner


def require_batch(key: str) -> dict | None:
    """Return the session batch, or render an empty state with two ways forward."""
    batch = state.get_batch()
    if batch:
        if batch["synthetic"]:
            synthetic_banner()
        return batch
    empty_state(t("batchui.empty.msg"), t("batchui.empty.detail"), icon="▦")
    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        navigation.link("batch", label=t("batchui.empty.go"), icon=":material/upload_file:")
    with c2:
        if st.button(t("batchui.empty.demo"), key=f"{key}_quick_demo", type="primary", icon=":material/bolt:"):
            bar = st.progress(0.0, text=t("batch.score.progress"))
            batchops.quick_demo(5000, progress=lambda f: bar.progress(f, text=t("batch.score.progress")))
            st.rerun()
    return None


def customer_360(row: pd.Series) -> None:
    """Profile card + probability + reasons + next best action for one scored customer."""
    bundle = artifacts.get_bundle()
    config = bundle["preprocessing_config"]
    profiles = artifacts.get_segment_profiles()
    pf = portfolio.compute(profiles, artifacts.get_metadata())
    raw = {c: row.get(c) for c in [config["id_column"]] + list(config["raw_source_columns"])}
    raw = {k: (np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else v) for k, v in raw.items()}
    res = scoring.score_one(raw, bundle)

    c1, c2, c3 = st.columns([1, 1.1, 1.1])
    with c1:
        prob_gauge(res["probability"], reference=pf.churn_rate / 100)
        kpi_row([
            dict(label=t("batchui.c360.decile"), value=str(int(row.get("risk_decile", 0))),
                 help=t("glossary.decile.def")),
            dict(label=t("cc.result.segment"), value=segment_name(res["segment"])),
        ], cols=1)
    with c2:
        rows = []
        for c in config["raw_source_columns"]:
            v = raw.get(c)
            shown = t("cc.form.not_recorded") if (v is None or (isinstance(v, float) and math.isnan(v))) else (
                fmt.num(v) if isinstance(v, (int, float, np.floating)) else str(v))
            rows.append(f"<tr><td style='opacity:.7;padding:.15rem .6rem .15rem 0'>{html.escape(friendly(c))}</td>"
                        f"<td style='font-weight:600'>{html.escape(shown)}</td></tr>")
        st.html(f'<div class="hub-card"><div class="hub-kpi-label">{html.escape(t("batchui.c360.profile"))}</div>'
                f'<table style="font-size:.86rem;margin-top:.3rem">{"".join(rows)}</table></div>')
    with c3:
        items = explain.reason_sentences(res["grouped"], raw, res["features"], config, n=4)
        lines = "".join(f'<div class="hub-reason"><span class="dir {d}">{"▲" if d == "up" else "▼"}</span>'
                        f'<span>{html.escape(s)}</span></div>' for d, s in items)
        st.html(f'<div class="hub-card"><div class="hub-kpi-label">{html.escape(t("cc.reasons.title"))}</div>{lines}</div>')
        ctx = explain.build_context(res["probability"], res["grouped"], raw, res["features"], pf.churn_rate / 100,
                                    profiles)
        for k in explain.next_best_actions(ctx, limit=2):
            insight_callout(t(f"nba.{k}.body"), "good" if k == "low_risk" else "info", tag=t(f"nba.{k}.title"))
        st.caption(t("cc.nba.caption"))
