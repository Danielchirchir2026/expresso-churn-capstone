"""Batch Scoring: get data, validate it, score in chunks, explore and export."""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import artifacts, batchops, demo, fmt, portfolio, state, validation
from app.core.explain import reason_label
from app.core.features import friendly
from app.core.segments import SEGMENT_IDS, segment_name
from app.i18n import t
from app.ui import charts
from app.ui.batchui import customer_360
from app.ui.components import insight_callout, kpi_row, page_header, section_header, synthetic_banner

bundle = artifacts.get_bundle()
config = bundle["preprocessing_config"]
ID = config["id_column"]
profiles = artifacts.get_segment_profiles()
pf = portfolio.compute(profiles, artifacts.get_metadata())

page_header(t("batch.title"), t("batch.sub"), t("batch.eyebrow"))


@st.cache_data(max_entries=3, show_spinner=False)
def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), low_memory=False)


# ---------------------------------------------------------------------------
# 1 · Get data
# ---------------------------------------------------------------------------
section_header(t("batch.step1.title"), t("batch.step1.sub"))
tab_up, tab_demo, tab_tpl = st.tabs([t("batch.tab.upload"), t("batch.tab.demo"), t("batch.tab.template")])
with tab_up:
    up = st.file_uploader(t("batch.upload.label"), type=["csv"], help=t("batch.upload.help"), key="batch_upload")
    if up is not None and st.session_state.get("batch_upload_id") != up.file_id:
        try:
            df_up = read_csv(up.getvalue())
            st.session_state["batch_pending"] = {"df": df_up, "name": up.name, "synthetic": False}
            st.session_state["batch_upload_id"] = up.file_id
        except Exception as exc:  # unreadable file -> friendly message
            st.error(t("batch.upload.unreadable", err=type(exc).__name__))
with tab_demo:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        n_demo = st.number_input(t("batch.demo.rows"), 100, 200_000, 5_000, 500, key="batch_demo_n")
    with c2:
        seed = st.number_input(t("batch.demo.seed"), 0, 10_000, 42, key="batch_demo_seed")
    with c3:
        messy = st.checkbox(t("batch.demo.messy"), value=False, key="batch_demo_messy", help=t("batch.demo.messy_help"))
    if st.button(t("batch.demo.button"), icon=":material/auto_awesome:", type="primary", key="batch_demo_go"):
        df_demo = demo.generate(int(n_demo), profiles, config, seed=int(seed),
                                breakdowns=artifacts.get_optional("eda_breakdowns"), messy=messy)
        st.session_state["batch_pending"] = {"df": df_demo, "name": f"synthetic-demo-{int(n_demo)}", "synthetic": True}
    st.caption(t("batch.demo.caption"))
with tab_tpl:
    st.markdown(t("batch.template.body"))
    st.code(", ".join([ID] + list(config["raw_source_columns"])), language=None, wrap_lines=True)
    st.download_button(t("batch.template.button"), validation.template(config).to_csv(index=False).encode("utf-8"),
                       file_name="expresso_scoring_template.csv", mime="text/csv", icon=":material/download:")

# ---------------------------------------------------------------------------
# 2 · Check  3 · Score
# ---------------------------------------------------------------------------
pending = st.session_state.get("batch_pending")
if pending:
    df_in = pending["df"]
    section_header(t("batch.step2.title", name=pending["name"]), t("batch.step2.sub"))
    if pending["synthetic"]:
        synthetic_banner()
    report = validation.validate(df_in, config)
    kpi_row([
        dict(label=t("batch.check.rows"), value=fmt.num(report.n_rows)),
        dict(label=t("batch.check.cols"), value=fmt.num(report.n_cols)),
        dict(label=t("batch.check.blocking"), value=str(len(report.blocking)),
             tone="bad" if report.blocking else "good", delta=t("batch.check.must_fix") if report.blocking else None),
        dict(label=t("batch.check.warnings"), value=str(len(report.warnings))),
    ], cols=4)
    if not report.issues:
        insight_callout(t("batch.check.all_good"), "good")
    tone = {"block": "risk", "warn": "warn", "info": "info"}
    for issue in report.issues:
        rows = (" " + t("batch.check.rows_list", rows=", ".join(map(str, issue.rows)),
                        more=f" (+{fmt.num(issue.count - len(issue.rows))})" if issue.count > len(issue.rows) else "")
                if issue.rows else "")
        insight_callout(t(f"batch.issue.{issue.code}", n=fmt.num(issue.count), **issue.params) + rows,
                        tone[issue.severity], tag=t(f"batch.severity.{issue.severity}"))
    with st.expander(t("batch.check.preview")):
        st.dataframe(df_in.head(20).astype("string"), hide_index=True, width="stretch")

    codes = {i.code for i in report.blocking}
    fixable = {"missing_ids", "duplicate_ids"}
    drop_ids = drop_dups = False
    if codes & fixable:
        c1, c2 = st.columns(2)
        with c1:
            if "missing_ids" in codes:
                drop_ids = st.checkbox(t("batch.fix.drop_missing_ids"), value=True, key="batch_fix_ids")
        with c2:
            if "duplicate_ids" in codes:
                drop_dups = st.checkbox(t("batch.fix.drop_dups"), value=True, key="batch_fix_dups")
    unresolved = codes - ({"missing_ids"} if drop_ids else set()) - ({"duplicate_ids"} if drop_dups else set())

    section_header(t("batch.step3.title"), t("batch.step3.sub"))
    n_rows = max(report.n_rows, 1)
    explain_n = st.number_input(t("batch.score.explain"), 0, n_rows, min(2000, n_rows), 500, key="batch_explain_n",
                                help=t("batch.score.explain_help"))
    if unresolved:
        st.error(t("batch.score.blocked"), icon=":material/block:")
    if st.button(t("batch.score.button", n=fmt.num(report.n_rows)), type="primary", icon=":material/rocket_launch:",
                 disabled=bool(unresolved), key="batch_score_go", width="stretch"):
        clean = validation.clean(df_in, config, drop_missing_ids=drop_ids, drop_duplicates=drop_dups)
        bar = st.progress(0.0, text=t("batch.score.progress"))
        batchops.score_and_store(clean, pending["name"], pending["synthetic"],
                                 progress=lambda f: bar.progress(f, text=t("batch.score.progress")),
                                 reasons_for=int(explain_n))
        bar.empty()
        st.session_state.pop("batch_pending", None)
        st.toast(t("batch.score.done", n=fmt.num(len(clean))), icon=":material/check_circle:")
        st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
batch = state.get_batch()
if not batch:
    if not pending:
        st.info(t("batch.results.none"), icon=":material/info:")
else:
    df = batch["df"]
    section_header(t("batch.results.title", name=batch["name"]),
                   t("batch.results.sub", n=fmt.num(batch["rows"]), at=batch["scored_at"]))
    if batch["synthetic"]:
        synthetic_banner()

    with st.container(border=True):
        st.markdown(f"**{t('batch.bands.title')}** · {t('ui.exploratory_note')}")
        c1, c2 = st.columns(2)
        with c1:
            low = st.slider(t("batch.bands.low"), 0.0, 1.0, float(round(pf.churn_rate / 100, 3)), 0.005,
                            key="band_low", format="%.3f", help=t("batch.bands.low_help"))
        with c2:
            high = st.slider(t("batch.bands.high"), 0.0, 1.0, 0.5, 0.005, key="band_high", format="%.3f",
                             help=t("batch.bands.high_help"))
        if high <= low:
            st.warning(t("batch.bands.order"))
    st.session_state["band_thresholds"] = (low, high)

    p = df["churn_probability"].to_numpy()
    band = np.where(p >= high, "high", np.where(p >= low, "medium", "low"))
    kpi_row([
        dict(label=t("batch.kpi.customers"), value=fmt.num(len(df))),
        dict(label=t("batch.kpi.mean_p"), value=fmt.pct(p.mean() * 100, 2),
             delta=t("cc.result.portfolio_is", rate=fmt.pct(pf.churn_rate, 2)), tone="neutral"),
        dict(label=t("batch.kpi.expected"), value=fmt.num(p.sum()), help=t("batch.kpi.expected_help")),
        dict(label=t("batch.kpi.above_high"), value=fmt.num((band == "high").sum()),
             foot=t("batch.kpi.share", share=fmt.pct((band == "high").mean() * 100, 1))),
    ], cols=4)

    # Filters
    section_header(t("batch.filters.title"))
    region_vals = df["REGION"].fillna(t("cc.form.not_recorded")).astype(str)
    tenure_order = [x for x in config["tenure_order"] if x in set(df["TENURE"].dropna().astype(str))]
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        pr = st.slider(t("batch.filters.prob"), 0.0, 100.0, (0.0, 100.0), 0.5, key="bf_prob", format="%.1f%%")
    with c2:
        regs = st.multiselect(t("batch.filters.region"), sorted(region_vals.unique()), key="bf_region",
                              placeholder=t("batch.filters.all"))
    with c3:
        tens = st.multiselect(t("batch.filters.tenure"), tenure_order, key="bf_tenure", placeholder=t("batch.filters.all"))
    with c4:
        segs = st.multiselect(t("batch.filters.segment"), SEGMENT_IDS, format_func=segment_name, key="bf_segment",
                              placeholder=t("batch.filters.all"))
    mask = (p * 100 >= pr[0]) & (p * 100 <= pr[1])
    if regs:
        mask &= region_vals.isin(regs).to_numpy()
    if tens:
        mask &= df["TENURE"].astype(str).isin(tens).to_numpy()
    if segs:
        mask &= df["segment"].isin(segs).to_numpy()
    view = df[mask]
    st.caption(t("batch.filters.showing", n=fmt.num(len(view)), total=fmt.num(len(df))))

    # Table
    reasons = [c for c in ("reason_1", "reason_2", "reason_3") if c in view.columns]
    labels = {g: reason_label(g) for c in reasons for g in view[c].dropna().unique()}
    top = view.sort_values("churn_probability", ascending=False).head(1000)
    table = pd.DataFrame({
        ID: top[ID],
        "p": top["churn_probability"] * 100,
        "decile": top["risk_decile"],
        "band": [t(f"batch.bands.{b}_label") for b in np.where(top["churn_probability"] >= high, "high",
                                                          np.where(top["churn_probability"] >= low, "medium", "low"))],
        "segment": top["segment"].map(segment_name),
        "reasons": top[reasons].apply(lambda r: " · ".join(labels.get(x, "") for x in r if isinstance(x, str)), axis=1),
        "REGION": top["REGION"].fillna("—"), "TENURE": top["TENURE"].fillna("—"), "REVENUE": top["REVENUE"],
    })
    st.dataframe(table, hide_index=True, width="stretch", height=380, column_config={
        ID: st.column_config.TextColumn(t("cc.form.customer_id")),
        "p": st.column_config.ProgressColumn(t("ui.gauge.label"), format="%.1f%%", min_value=0, max_value=100),
        "decile": st.column_config.NumberColumn(t("batchui.c360.decile"), help=t("glossary.decile.def")),
        "band": st.column_config.TextColumn(t("batch.bands.col"), help=t("glossary.exploratory_band.def")),
        "segment": st.column_config.TextColumn(t("cc.result.segment")),
        "reasons": st.column_config.TextColumn(t("batch.table.reasons"), width="large"),
        "REGION": st.column_config.TextColumn(friendly("REGION")),
        "TENURE": st.column_config.TextColumn(friendly("TENURE")),
        "REVENUE": st.column_config.NumberColumn(friendly("REVENUE"), format="localized"),
    })
    if len(view) > 1000:
        st.caption(t("batch.table.capped"))
    if "reasons_computed" in df.columns and not df["reasons_computed"].all():
        st.caption(t("batch.table.reasons_partial", n=fmt.num(int(df["reasons_computed"].sum()))))

    # Charts
    if len(view):
        c1, c2 = st.columns(2)
        with c1:
            hist = go.Figure(go.Histogram(x=view["churn_probability"] * 100, xbins=dict(start=0, end=100, size=2),
                                          marker_color=charts.accent(),
                                          hovertemplate="%{x}%: %{y:,}<extra></extra>"))
            for x, name in ((low * 100, t("batch.bands.low")), (high * 100, t("batch.bands.high"))):
                hist.add_vline(x=x, line=dict(color=charts.ink("muted"), width=1),
                               annotation_text=name, annotation_font=dict(size=10, color=charts.ink("secondary")))
            charts.style(hist, height=340, legend=False, pct_x=True, title=t("batch.charts.hist"))
            hist.update_layout(bargap=0.05)
            hist.update_xaxes(title=t("ui.gauge.label"))
            hist.update_yaxes(title=t("batch.kpi.customers"))
            share_top = (view["churn_probability"] >= high).mean() * 100
            charts.show(hist, t("batch.charts.hist_so_what", share=fmt.pct(share_top, 1), high=fmt.pct(high * 100, 1),
                                median=fmt.pct(view["churn_probability"].median() * 100, 1)), key="b_hist")
        with c2:
            reg = (view.assign(r=region_vals[mask].to_numpy()).groupby("r")["churn_probability"]
                   .agg(["mean", "size"]).sort_values("mean"))
            fig = go.Figure(go.Bar(y=reg.index, x=reg["mean"] * 100, orientation="h", marker_color=charts.accent(),
                                   customdata=reg["size"], text=[fmt.pct(v * 100, 1) for v in reg["mean"]],
                                   textposition="outside",
                                   hovertemplate="%{y}: %{x:.2f}%<br>n = %{customdata:,}<extra></extra>"))
            charts.style(fig, height=340, legend=False, pct_x=True, xgrid=True, ygrid=False,
                         title=t("batch.charts.region"))
            fig.update_xaxes(range=[0, reg["mean"].max() * 130])
            charts.show(fig, t("batch.charts.region_so_what", top=reg.index[-1], p=fmt.pct(reg["mean"].iloc[-1] * 100, 1),
                               low=reg.index[0], lp=fmt.pct(reg["mean"].iloc[0] * 100, 1)),
                        data=reg.reset_index().rename(columns={"r": friendly("REGION"), "mean": t("batch.charts.mean_p"),
                                                               "size": t("batch.kpi.customers")}).round(4),
                        key="b_region")
        c1, c2 = st.columns(2)
        with c1:
            ten = view.groupby("TENURE")["churn_probability"].agg(["mean", "size"]).reindex(tenure_order).dropna()
            fig = go.Figure(go.Bar(x=ten.index, y=ten["mean"] * 100, marker_color=charts.accent(),
                                   customdata=ten["size"],
                                   hovertemplate="%{x}: %{y:.2f}%<br>n = %{customdata:,}<extra></extra>"))
            charts.style(fig, height=320, legend=False, pct_y=True, title=t("batch.charts.tenure"))
            if len(ten):
                charts.show(fig, t("batch.charts.tenure_so_what", top=ten["mean"].idxmax(),
                                   p=fmt.pct(ten["mean"].max() * 100, 1)),
                            data=ten.reset_index().rename(columns={"mean": t("batch.charts.mean_p"),
                                                                   "size": t("batch.kpi.customers")}).round(4),
                            key="b_tenure")
        with c2:
            sg = view.groupby("segment")["churn_probability"].agg(["mean", "size"]).reindex(SEGMENT_IDS).dropna()
            fig = go.Figure(go.Bar(x=[segment_name(s) for s in sg.index], y=sg["mean"] * 100,
                                   marker_color=[charts.segment_color(s) for s in sg.index], customdata=sg["size"],
                                   hovertemplate="%{x}: %{y:.2f}%<br>n = %{customdata:,}<extra></extra>"))
            charts.style(fig, height=320, legend=False, pct_y=True, title=t("batch.charts.segment"))
            if len(sg):
                charts.show(fig, t("batch.charts.segment_so_what", top=segment_name(sg["mean"].idxmax()),
                                   p=fmt.pct(sg["mean"].max() * 100, 1),
                                   share=fmt.pct(sg["size"].max() / sg["size"].sum() * 100, 0),
                                   big=segment_name(sg["size"].idxmax())), key="b_segment")

    # Customer 360
    section_header(t("batch.c360.title"), t("batch.c360.sub"))
    riskiest = df.nlargest(50, "churn_probability")[ID].astype(str).tolist()
    c1, c2 = st.columns([1, 1])
    with c1:
        pick = st.selectbox(t("batch.c360.pick"), riskiest, key="c360_pick")
    with c2:
        typed = st.text_input(t("batch.c360.search"), key="c360_search", placeholder=t("batch.c360.placeholder"))
    target = typed.strip() or pick
    hit = df[df[ID].astype(str) == target]
    if hit.empty:
        st.warning(t("batch.c360.not_found", id=target))
    else:
        customer_360(hit.iloc[0])

    # Downloads
    section_header(t("batch.dl.title"), t("batch.dl.sub"))
    export = df.assign(
        exploratory_band=np.where(p >= high, "High", np.where(p >= low, "Medium", "Low")),
        segment_name=df["segment"].map(lambda s: segment_name(s, "en")),
        **{f"{c}_label": df[c].map(lambda g: friendly(g, "en") if isinstance(g, str) else "") for c in reasons},
    )
    export = export.drop(columns=[c for c in export.columns if c.endswith("_logodds")])
    name = batch["name"].rsplit(".", 1)[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(t("batch.dl.csv"), data=lambda: export.to_csv(index=False).encode("utf-8"),
                           file_name=f"{name}_scored.csv", mime="text/csv", icon=":material/download:",
                           width="stretch", on_click="ignore")

    def to_excel(frame: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            frame.head(1_000_000).to_excel(xw, index=False, sheet_name="scored")
        return buf.getvalue()

    with c2:
        st.download_button(t("batch.dl.xlsx"), data=lambda: to_excel(export),
                           file_name=f"{name}_scored.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/table_view:", width="stretch", on_click="ignore",
                           help=t("batch.dl.xlsx_help"))
    with c3:
        k = st.number_input(t("batch.dl.k"), 1, max(len(df), 1), min(500, len(df)), 50, key="call_k")
    calls = export.nlargest(int(k), "churn_probability")
    st.download_button(t("batch.dl.calls", k=fmt.num(k)), data=lambda: calls.to_csv(index=False).encode("utf-8"),
                       file_name=f"{name}_priority_call_list_top{int(k)}.csv", mime="text/csv",
                       icon=":material/call:", type="primary", on_click="ignore")
    if st.button(t("batch.dl.clear"), icon=":material/delete:", key="batch_clear"):
        state.clear_batch()
        st.rerun()
