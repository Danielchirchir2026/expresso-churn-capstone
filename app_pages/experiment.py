"""Experiment Designer: plan an A/B test of a retention offer and analyse the result."""

import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, fmt, metrics, portfolio, state
from app.core.segments import segment_name
from app.i18n import t
from app.ui import charts
from app.ui.components import insight_callout, kpi_row, page_header, section_header, synthetic_banner

pf = portfolio.compute(artifacts.get_segment_profiles(), artifacts.get_metadata())
batch = state.get_batch()
ID = artifacts.get_config()["id_column"]

page_header(t("exp.title"), t("exp.sub"), t("exp.eyebrow"))
tab_plan, tab_analyse = st.tabs([t("exp.tab.plan"), t("exp.tab.analyse")])

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
with tab_plan:
    section_header(t("exp.plan.title"), t("exp.plan.sub"))
    if batch:
        k_default = min(len(batch["df"]), 2000)
        base_default = float(batch["df"].nlargest(k_default, "churn_probability")["churn_probability"].mean() * 100)
        base_help = t("exp.plan.base_help_batch", k=fmt.num(k_default))
    else:
        base_default = float(pf.churn_rate)
        base_help = t("exp.plan.base_help_portfolio")
    c1, c2, c3 = st.columns(3)
    with c1:
        base = st.number_input(t("exp.plan.base"), 0.5, 99.0, round(base_default, 2), 0.5, key="exp_base",
                               help=base_help)
        st.session_state.setdefault("exp_mode", "relative")
        mode = st.segmented_control(t("exp.plan.mode"), ["relative", "absolute"], key="exp_mode",
                                    format_func=lambda m: t(f"exp.plan.mode_{m}")) or "relative"
        red = st.number_input(t(f"exp.plan.red_{mode}"), 0.1, 90.0, 20.0 if mode == "relative" else 3.0, 0.5,
                              key=f"exp_red_{mode}", help=t("exp.plan.red_help"))
    with c2:
        alpha = st.select_slider(t("exp.plan.alpha"), [0.01, 0.05, 0.10], value=0.05, key="exp_alpha",
                                 help=t("exp.plan.alpha_help"))
        power = st.select_slider(t("exp.plan.power"), [0.7, 0.8, 0.9, 0.95], value=0.8, key="exp_power",
                                 help=t("exp.plan.power_help"))
        two_sided = st.toggle(t("exp.plan.two_sided"), value=True, key="exp_two", help=t("exp.plan.two_sided_help"))
    with c3:
        weekly = st.number_input(t("exp.plan.weekly"), 1, 1_000_000, 500, 50, key="exp_weekly",
                                 help=t("exp.plan.weekly_help"))
    p1 = base / 100
    p2 = p1 * (1 - red / 100) if mode == "relative" else p1 - red / 100
    if not 0 < p2 < p1:
        st.error(t("exp.plan.bad_effect"), icon=":material/error:")
    else:
        n = metrics.sample_size_two_proportions(p1, p2, alpha, power, two_sided)
        weeks = math.ceil(n / weekly)
        kpi_row([
            dict(label=t("exp.plan.n_arm"), value=fmt.num(n), help=t("exp.plan.n_arm_help")),
            dict(label=t("exp.plan.total"), value=fmt.num(2 * n)),
            dict(label=t("exp.plan.target"), value=fmt.pct(p2 * 100, 2),
                 foot=t("exp.plan.target_foot", base=fmt.pct(p1 * 100, 2))),
            dict(label=t("exp.plan.weeks"), value=fmt.num(weeks), foot=t("exp.plan.weeks_foot", w=fmt.num(weekly))),
        ], cols=4)
        insight_callout(t("exp.plan.explain", n=fmt.num(n), base=fmt.pct(p1 * 100, 1), target=fmt.pct(p2 * 100, 1),
                          alpha=fmt.num(alpha, 2), power=fmt.pct(power * 100, 0), weeks=fmt.num(weeks)), "info")
        st.caption(t("exp.plan.formula"))

        # Split the priority call list
        section_header(t("exp.split.title"), t("exp.split.sub"))
        if not batch:
            st.info(t("exp.split.no_batch"), icon=":material/info:")
        else:
            if batch["synthetic"]:
                synthetic_banner()
            df = batch["df"]
            c1, c2 = st.columns(2)
            with c1:
                k = st.number_input(t("exp.split.k"), 2, len(df), min(len(df), 2 * n), 100, key="exp_k",
                                    help=t("exp.split.k_help"))
            with c2:
                seed = st.number_input(t("exp.split.seed"), 0, 1_000_000, 42, key="exp_seed")
            calls = df.nlargest(int(k), "churn_probability")[[ID, "churn_probability", "segment"]].copy()
            rng = np.random.default_rng(int(seed))
            groups = np.array(["treatment"] * (len(calls) // 2) + ["control"] * (len(calls) - len(calls) // 2))
            rng.shuffle(groups)
            calls["group"] = groups
            calls["segment_name"] = calls["segment"].map(lambda s: segment_name(s, "en"))
            per_arm = int(min((groups == "treatment").sum(), (groups == "control").sum()))
            achieved = metrics.power_two_proportions(p1, p2, per_arm, alpha, two_sided)
            summary = calls.groupby("group")["churn_probability"].agg(["size", "mean"])
            kpi_row([
                dict(label=t("exp.split.treatment"), value=fmt.num(summary.loc["treatment", "size"]),
                     foot=t("exp.split.mean_p", p=fmt.pct(summary.loc["treatment", "mean"] * 100, 2))),
                dict(label=t("exp.split.control"), value=fmt.num(summary.loc["control", "size"]),
                     foot=t("exp.split.mean_p", p=fmt.pct(summary.loc["control", "mean"] * 100, 2))),
                dict(label=t("exp.split.power"), value=fmt.pct(achieved * 100, 0),
                     tone="good" if achieved >= power else "bad",
                     delta=t("exp.split.power_target", p=fmt.pct(power * 100, 0))),
            ], cols=3)
            if per_arm < n:
                insight_callout(t("exp.split.underpowered", have=fmt.num(per_arm), need=fmt.num(n)), "warn")
            st.download_button(t("exp.split.download"),
                               data=lambda: calls.drop(columns=["segment"]).to_csv(index=False).encode("utf-8"),
                               file_name=f"ab_test_split_seed{int(seed)}.csv", mime="text/csv", type="primary",
                               icon=":material/download:", on_click="ignore", key="exp_dl")
            st.caption(t("exp.split.caption"))

# ---------------------------------------------------------------------------
# Analyse results
# ---------------------------------------------------------------------------
with tab_analyse:
    section_header(t("exp.res.title"), t("exp.res.sub"))
    up = st.file_uploader(t("exp.res.upload"), type=["csv"], key="exp_up", help=t("exp.res.upload_help"))
    for key, val in {"nt": 1000, "ct": 150, "nc": 1000, "cc": 200}.items():  # example values, editable
        st.session_state.setdefault(f"exp_res_{key}", val)
    if up is not None and st.session_state.get("exp_up_id") != up.file_id:
        try:
            res = pd.read_csv(io.BytesIO(up.getvalue()))
            res["group"] = res["group"].astype(str).str.strip().str.lower()
            r = res.set_index("group")
            defaults = {"nt": int(r.loc["treatment", "customers"]), "ct": int(r.loc["treatment", "churned"]),
                        "nc": int(r.loc["control", "customers"]), "cc": int(r.loc["control", "churned"])}
            for key, val in defaults.items():
                st.session_state[f"exp_res_{key}"] = val
            st.session_state["exp_up_id"] = up.file_id
        except Exception:
            st.error(t("exp.res.bad_file"), icon=":material/error:")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{t('exp.split.treatment')}**")
        nt = st.number_input(t("exp.res.customers"), 1, 10_000_000, key="exp_res_nt")
        ct = st.number_input(t("exp.res.churned"), 0, 10_000_000, key="exp_res_ct")
    with c2:
        st.markdown(f"**{t('exp.split.control')}**")
        nc = st.number_input(t("exp.res.customers"), 1, 10_000_000, key="exp_res_nc")
        cc = st.number_input(t("exp.res.churned"), 0, 10_000_000, key="exp_res_cc")
    st.caption(t("exp.res.example"))
    if ct > nt or cc > nc:
        st.error(t("exp.res.bad_counts"), icon=":material/error:")
    else:
        alpha_r = st.session_state.get("exp_alpha", 0.05)
        r = metrics.two_proportion_test(int(ct), int(nt), int(cc), int(nc), alpha_r)
        kpi_row([
            dict(label=t("exp.res.rate_t"), value=fmt.pct(r["rate_treatment"] * 100, 2)),
            dict(label=t("exp.res.rate_c"), value=fmt.pct(r["rate_control"] * 100, 2)),
            dict(label=t("exp.res.uplift"), value=fmt.pp(r["uplift"] * 100, 2),
                 tone="good" if r["uplift"] > 0 else "bad",
                 delta=t("exp.res.relative", rel=fmt.pct(r["relative_reduction"] * 100, 0))),
            dict(label=t("exp.res.ci", level=fmt.pct((1 - alpha_r) * 100, 0)),
                 value=f"{fmt.pp(r['ci_low'] * 100, 2)} … {fmt.pp(r['ci_high'] * 100, 2)}"),
            dict(label=t("exp.res.p"), value=fmt.num(r["p_value"], 4)),
        ], cols=5)
        if r["significant"] and r["uplift"] > 0:
            tone, key = "good", "exp.res.verdict_win"
        elif r["significant"]:
            tone, key = "risk", "exp.res.verdict_harm"
        else:
            tone, key = "warn", "exp.res.verdict_none"
        insight_callout(t(key, uplift=fmt.num(abs(r["uplift"]) * 100, 2), p=fmt.num(r["p_value"], 4),
                          lo=fmt.num(r["ci_low"] * 100, 2), hi=fmt.num(r["ci_high"] * 100, 2),
                          saved=fmt.num(max(r["uplift"], 0) * nt, 0)), tone, tag=t("exp.res.verdict_tag"))
        se = [math.sqrt(r["rate_treatment"] * (1 - r["rate_treatment"]) / nt),
              math.sqrt(r["rate_control"] * (1 - r["rate_control"]) / nc)]
        fig = go.Figure(go.Bar(x=[t("exp.split.treatment"), t("exp.split.control")],
                               y=[r["rate_treatment"] * 100, r["rate_control"] * 100],
                               marker_color=[charts.accent(), charts.neutral()],
                               error_y=dict(type="data", array=[1.96 * s * 100 for s in se], color=charts.ink("secondary")),
                               hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
        charts.style(fig, height=320, legend=False, pct_y=True)
        charts.show(fig, t("exp.res.so_what", t=fmt.pct(r["rate_treatment"] * 100, 1),
                           c=fmt.pct(r["rate_control"] * 100, 1)), key="exp_res_chart")
