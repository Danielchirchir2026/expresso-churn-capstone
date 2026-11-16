"""Campaign Tracker: log campaigns and compare realised results with the plan."""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import navigation
from app.core import fmt
from app.i18n import t
from app.ui import charts
from app.ui.components import empty_state, insight_callout, kpi_row, page_header, section_header

COLUMNS = ["campaign", "segment", "contacted", "offer", "cost_fcfa", "retained", "value_per_retained_fcfa",
           "planned_retained", "planned_net_fcfa"]
NUMERIC = ["contacted", "cost_fcfa", "retained", "value_per_retained_fcfa", "planned_retained", "planned_net_fcfa"]

page_header(t("camp.title"), t("camp.sub"), t("camp.eyebrow"))


def empty_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64" if c in NUMERIC else "string") for c in COLUMNS})


st.session_state.setdefault("campaign_log", empty_log())
st.session_state.setdefault("camp_v", 0)


def set_log(df: pd.DataFrame) -> None:
    df = df.reindex(columns=COLUMNS)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    st.session_state["campaign_log"] = df.reset_index(drop=True)
    st.session_state["camp_v"] += 1  # fresh editor so edits are never applied twice


def apply_edits() -> None:
    key = f"camp_editor_{st.session_state['camp_v']}"
    delta = st.session_state.get(key, {})
    df = st.session_state["campaign_log"].copy()
    for idx, changes in delta.get("edited_rows", {}).items():
        for col, val in changes.items():
            df.loc[int(idx), col] = val
    if delta.get("added_rows"):
        df = pd.concat([df, pd.DataFrame(delta["added_rows"])], ignore_index=True)
    if delta.get("deleted_rows"):
        df = df.drop(index=[int(i) for i in delta["deleted_rows"]], errors="ignore")
    set_log(df)


def add_plan() -> None:
    plan = st.session_state.get("ret_last_plan")
    if not plan:
        return
    log = st.session_state["campaign_log"]
    row = {"campaign": t("camp.add.name", n=len(log) + 1), "segment": t("camp.add.segment", batch=plan["batch"]),
           "contacted": plan["k"], "offer": t("camp.add.offer", cost=fmt.num(plan["offer_cost"])),
           "cost_fcfa": plan["k"] * plan["offer_cost"], "retained": np.nan,
           "value_per_retained_fcfa": round(plan["value_per_saved"]), "planned_retained": round(plan["expected_saved"], 1),
           "planned_net_fcfa": round(plan["expected_net"])}
    set_log(pd.concat([log, pd.DataFrame([row])], ignore_index=True))


# ---------------------------------------------------------------------------
# Add / import / export
# ---------------------------------------------------------------------------
section_header(t("camp.log.title"), t("camp.log.sub"))
plan = st.session_state.get("ret_last_plan")
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.button(t("camp.add.button"), icon=":material/playlist_add:", on_click=add_plan, disabled=not plan,
              key="camp_add", type="primary", help=t("camp.add.help") if plan else t("camp.add.no_plan"))
    if not plan:
        navigation.link("retention", label=t("camp.add.go_planner"))
    elif plan.get("synthetic"):
        st.caption(t("camp.add.synthetic"))
with c2:
    up = st.file_uploader(t("camp.io.import"), type=["csv"], key="camp_up", help=t("camp.io.import_help"),
                          label_visibility="visible")
    if up is not None and st.session_state.get("camp_up_id") != up.file_id:
        try:
            imported = pd.read_csv(io.BytesIO(up.getvalue()))
            missing = [c for c in COLUMNS if c not in imported.columns]
            if missing:
                st.error(t("camp.io.bad", cols=", ".join(missing)), icon=":material/error:")
            else:
                set_log(imported)
                st.session_state["camp_up_id"] = up.file_id
                st.rerun()
        except Exception:
            st.error(t("camp.io.unreadable"), icon=":material/error:")
log = st.session_state["campaign_log"]
with c3:
    st.download_button(t("camp.io.export"), log.to_csv(index=False).encode("utf-8"), "campaign_log.csv", "text/csv",
                       icon=":material/download:", key="camp_dl", disabled=log.empty, width="stretch")

edited = st.data_editor(
    log, key=f"camp_editor_{st.session_state['camp_v']}", on_change=apply_edits, num_rows="dynamic",
    hide_index=True, width="stretch", column_config={
        "campaign": st.column_config.TextColumn(t("camp.col.campaign"), required=True),
        "segment": st.column_config.TextColumn(t("camp.col.segment")),
        "contacted": st.column_config.NumberColumn(t("camp.col.contacted"), min_value=0, step=1, format="%d"),
        "offer": st.column_config.TextColumn(t("camp.col.offer")),
        "cost_fcfa": st.column_config.NumberColumn(t("camp.col.cost"), min_value=0, format="localized"),
        "retained": st.column_config.NumberColumn(t("camp.col.retained"), min_value=0, step=1, format="%d",
                                                  help=t("camp.col.retained_help")),
        "value_per_retained_fcfa": st.column_config.NumberColumn(t("camp.col.value"), min_value=0, format="localized",
                                                                 help=t("camp.col.value_help")),
        "planned_retained": st.column_config.NumberColumn(t("camp.col.planned_retained"), format="%.1f"),
        "planned_net_fcfa": st.column_config.NumberColumn(t("camp.col.planned_net"), format="localized"),
    })
st.caption(t("camp.log.caption"))

# ---------------------------------------------------------------------------
# Realised vs planned
# ---------------------------------------------------------------------------
section_header(t("camp.res.title"), t("camp.res.sub"))
done = log.dropna(subset=["retained", "contacted"])
done = done[done["contacted"] > 0]
if done.empty:
    empty_state(t("camp.res.empty"), t("camp.res.empty_detail"), icon="▦")
else:
    d = done.copy()
    d["value"] = d["retained"] * d["value_per_retained_fcfa"].fillna(0)
    d["net"] = d["value"] - d["cost_fcfa"].fillna(0)
    d["roi"] = np.where(d["cost_fcfa"] > 0, d["net"] / d["cost_fcfa"] * 100, np.nan)
    d["kept_per_100"] = d["retained"] / d["contacted"] * 100
    d["planned_per_100"] = d["planned_retained"] / d["contacted"] * 100
    cost, value = float(d["cost_fcfa"].sum()), float(d["value"].sum())
    net = value - cost
    kpi_row([
        dict(label=t("camp.res.contacted"), value=fmt.num(d["contacted"].sum())),
        dict(label=t("camp.res.retained"), value=fmt.num(d["retained"].sum()),
             foot=t("camp.res.per100", n=fmt.num(d["retained"].sum() / d["contacted"].sum() * 100, 1))),
        dict(label=t("camp.res.net"), value=fmt.money(net), tone="good" if net > 0 else "bad",
             delta=t("ret.kpi.roi", roi=fmt.pct(net / cost * 100, 0)) if cost else None),
        dict(label=t("camp.res.vs_plan"), value=fmt.money(net - float(d["planned_net_fcfa"].fillna(0).sum())),
             help=t("camp.res.vs_plan_help")),
    ], cols=4)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["campaign"], y=d["planned_net_fcfa"], name=t("camp.res.planned"),
                         marker_color=charts.neutral(), hovertemplate="%{x}: %{y:,.0f} FCFA<extra></extra>"))
    fig.add_trace(go.Bar(x=d["campaign"], y=d["net"], name=t("camp.res.realised"), marker_color=charts.accent(),
                         hovertemplate="%{x}: %{y:,.0f} FCFA<extra></extra>"))
    charts.style(fig, height=360)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=t("camp.res.axis"))
    best = d.loc[d["net"].idxmax()]
    charts.show(fig, t("camp.res.so_what", camp=best["campaign"], net=fmt.money(best["net"]),
                       roi=fmt.pct(best["roi"], 0) if np.isfinite(best["roi"]) else "–"),
                data=d[["campaign", "contacted", "retained", "kept_per_100", "planned_per_100", "net",
                        "planned_net_fcfa", "roi"]].round(2), key="camp_chart")
    for _, r in d.iterrows():
        if np.isfinite(r["planned_per_100"]):
            tone = "good" if r["kept_per_100"] >= r["planned_per_100"] else "warn"
            insight_callout(t("camp.res.row", camp=r["campaign"], real=fmt.num(r["kept_per_100"], 1),
                              plan=fmt.num(r["planned_per_100"], 1)), tone)
