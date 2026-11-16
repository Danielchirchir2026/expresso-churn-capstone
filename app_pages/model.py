"""Model & Explainability: model card, global drivers, how reason codes work."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import navigation
from app.core import artifacts, fmt
from app.core.features import (duplicated_indicators, friendly, global_importance, grouped_importance,
                               transformation)
from app.i18n import t
from app.ui import charts
from app.ui.components import insight_callout, kpi_row, page_header, section_header

bundle = artifacts.get_bundle()
model = bundle["model"]
config = bundle["preprocessing_config"]
meta = artifacts.get_metadata()
params = model.get_params()
n_trees = model.booster_.num_trees()
feat_cols = list(config["feature_columns"])
raw_cols = list(config["raw_source_columns"])

page_header(t("model.title"), t("model.sub"), t("model.eyebrow"))

# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
section_header(t("model.card.title"), t("model.card.sub"))
kpi_row([
    dict(label=t("model.card.type"), value=f"{meta['model_name']}", foot=type(model).__name__),
    dict(label=t("kpi.roc_auc"), value=fmt.num(meta["validation_roc_auc"], 4),
         foot=t("model.card.seed", seed=meta["validation_split_seed"])),
    dict(label=t("model.card.features"), value=str(meta["feature_count"]),
         foot=t("model.card.from_raw", n=len(raw_cols))),
    dict(label=t("model.card.trees"), value=str(n_trees),
         foot=t("model.card.best_iter", n=meta["best_iteration"])),
    dict(label=t("model.card.rows"), value=fmt.num(meta["training_rows"])),
    dict(label=t("model.card.thresholds"), value=t("model.card.yes") if meta["risk_thresholds_approved"]
         else t("model.card.no"), tone="bad" if not meta["risk_thresholds_approved"] else "good",
         delta=t("model.card.probabilities_only") if not meta["risk_thresholds_approved"] else None),
], cols=3)

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown(f"**{t('model.card.intended_title')}**")
        for i in range(1, 4):
            st.markdown(f"- {t(f'model.card.intended_{i}')}")
        st.markdown(f"**{t('model.card.not_for_title')}**")
        for i in range(1, 3):
            st.markdown(f"- {t(f'model.card.not_for_{i}')}")
with c2:
    hp = pd.DataFrame({
        t("model.card.param"): ["num_leaves", "learning_rate", "n_estimators", "min_child_samples", "subsample",
                                "colsample_bytree", "random_state"],
        t("model.card.value"): [str(params.get(k)) for k in ["num_leaves", "learning_rate", "n_estimators",
                                                            "min_child_samples", "subsample", "colsample_bytree",
                                                            "random_state"]],
    })
    versions = pd.DataFrame({
        t("model.card.param"): ["Python", "LightGBM", "scikit-learn", "pandas", "NumPy", t("model.card.created")],
        t("model.card.value"): [meta["python_version"], meta["lightgbm_version"], meta["scikit_learn_version"],
                                meta["pandas_version"], meta["numpy_version"], meta["created_utc"][:19] + " UTC"],
    })
    st.dataframe(pd.concat([hp, versions], ignore_index=True), hide_index=True, width="stretch", height=455)

section_header(t("model.limits.title"), t("model.limits.sub"))
dups = duplicated_indicators(feat_cols)
insight_callout(t("model.limits.duplicates", n=len(dups), example=f"{dups[0][0]} / {dups[0][1]}" if dups else "–"),
                "warn")
if n_trees > int(meta["best_iteration"]) and int(meta["training_rows"]) > 0:
    insight_callout(t("model.limits.refit", trees=n_trees, best=meta["best_iteration"],
                      rows=fmt.num(meta["training_rows"])), "warn")
insight_callout(t("model.limits.region_missing"), "info")
insight_callout(t("model.limits.snapshot"), "info")
insight_callout(t("model.limits.thresholds"), "info")

# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
section_header(t("model.imp.title"), t("model.imp.sub"))
imp = global_importance(model, tuple(feat_cols), tuple(raw_cols))
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.session_state.setdefault("imp_kind", "gain")
    kind = st.segmented_control(t("model.imp.kind"), ["gain", "split"], key="imp_kind",
                                format_func=lambda k: t(f"model.imp.kind_{k}"), help=t("model.imp.kind_help"))
    kind = kind or "gain"
with c2:
    grouped = st.toggle(t("model.imp.group"), value=True, key="imp_group", help=t("model.imp.group_help"))
with c3:
    top_n = st.slider(t("model.imp.top"), 5, 30, 12, key="imp_top")
share_col = f"{kind}_share"
if grouped:
    table = grouped_importance(imp).sort_values(share_col, ascending=False)
    table["label"] = table["group"].map(friendly)
else:
    table = imp.sort_values(share_col, ascending=False).copy()
    table["label"] = table["feature"]
top = table.head(top_n).iloc[::-1]
fig = go.Figure(go.Bar(y=top["label"], x=top[share_col], orientation="h", marker_color=charts.accent(),
                       text=[fmt.pct(v, 1) for v in top[share_col]], textposition="outside",
                       hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>"))
charts.style(fig, height=max(320, 28 * len(top) + 80), legend=False, pct_x=True, xgrid=True, ygrid=False)
fig.update_xaxes(title=t(f"model.imp.axis_{kind}"), range=[0, top[share_col].max() * 1.18])
first, second = table.iloc[0], table.iloc[1]
charts.show(fig, t(f"model.imp.so_what_{kind}", a=first["label"], b=second["label"],
                   share=fmt.pct(first[share_col] + second[share_col], 0)),
            data=table[["label", share_col]].rename(columns={"label": t("model.imp.col_feature"),
                                                             share_col: t(f"model.imp.axis_{kind}")}).round(3),
            key="model_imp")
gain_top = grouped_importance(imp).iloc[0]
split_top = grouped_importance(imp).sort_values("split", ascending=False).iloc[0]
if gain_top["group"] != split_top["group"]:
    insight_callout(t("model.imp.gain_vs_split", gain=friendly(gain_top["group"]), split=friendly(split_top["group"])),
                    "info")

with st.expander(t("model.imp.all_features"), icon=":material/list:"):
    full = imp.assign(source=imp["group"].map(friendly), how=imp["feature"].map(transformation))
    st.dataframe(full[["feature", "source", "how", "gain_share", "split"]], hide_index=True, width="stretch",
                 column_config={
                     "feature": st.column_config.TextColumn(t("model.imp.col_feature")),
                     "source": st.column_config.TextColumn(t("exec.drivers.col_source")),
                     "how": st.column_config.TextColumn(t("model.imp.col_how"), width="large"),
                     "gain_share": st.column_config.NumberColumn(t("model.imp.axis_gain"), format="%.2f%%"),
                     "split": st.column_config.NumberColumn(t("model.imp.col_split")),
                 })

# ---------------------------------------------------------------------------
# How reason codes work
# ---------------------------------------------------------------------------
section_header(t("model.reasons.title"), t("model.reasons.sub"))
for i in range(1, 4):
    st.markdown(f"{i}. {t(f'model.reasons.step_{i}')}")
c1, c2 = st.columns(2)
with c1:
    navigation.link("customer_check", label=t("model.reasons.try"))
with c2:
    navigation.link("evaluation", label=t("model.reasons.eval"))
