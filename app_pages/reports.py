"""Reports: executive brief, batch outputs, reference files."""

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from app import navigation
from app.core import artifacts, fmt, state
from app.core.dictionary import data_dictionary
from app.core.features import friendly
from app.core.pdf import executive_brief
from app.core.segments import segment_name
from app.i18n import current_lang, t
from app.ui.components import empty_state, page_header, section_header, synthetic_banner

lang = current_lang()
meta = artifacts.get_metadata()
config = artifacts.get_config()
profiles = artifacts.get_segment_profiles()
features = artifacts.get_feature_columns()
batch = state.get_batch()

page_header(t("rep.title"), t("rep.sub"), t("rep.eyebrow"))


def card(title: str, body: str):
    box = st.container(border=True)
    box.markdown(f"**{title}**")
    box.caption(body)
    return box


# ---------------------------------------------------------------------------
# Briefs
# ---------------------------------------------------------------------------
section_header(t("rep.brief.title"), t("rep.brief.sub"))
with card(t("rep.brief.pdf"), t("rep.brief.pdf_desc")):
    st.download_button(t("rep.brief.download"), data=lambda: executive_brief(profiles, meta, lang, batch),
                       file_name=f"expresso_executive_brief_{lang}.pdf", mime="application/pdf",
                       icon=":material/picture_as_pdf:", type="primary", on_click="ignore", key="rep_pdf")

# ---------------------------------------------------------------------------
# Batch outputs
# ---------------------------------------------------------------------------
section_header(t("rep.batch.title"), t("rep.batch.sub"))
if not batch:
    empty_state(t("batchui.empty.msg"), t("rep.batch.empty"), icon="▦")
    navigation.link("batch", label=t("batchui.empty.go"))
else:
    if batch["synthetic"]:
        synthetic_banner()
    df = batch["df"]
    low, high = st.session_state.get("band_thresholds", (None, None))
    export = df.drop(columns=[c for c in df.columns if c.endswith("_logodds")]).assign(
        segment_name=df["segment"].map(lambda s: segment_name(s, "en")),
        **{f"{c}_label": df[c].map(lambda g: friendly(g, "en") if isinstance(g, str) else "")
           for c in ("reason_1", "reason_2", "reason_3") if c in df.columns})
    if low is not None:
        p = export["churn_probability"]
        export["exploratory_band"] = np.where(p >= high, "High", np.where(p >= low, "Medium", "Low"))
    base = batch["name"].rsplit(".", 1)[0]

    def xlsx() -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            export.head(1_000_000).to_excel(xw, index=False, sheet_name="scored")
        return buf.getvalue()

    c1, c2, c3 = st.columns(3)
    with c1:
        with card(t("rep.batch.csv"), t("rep.batch.rows", n=fmt.num(len(df)))):
            st.download_button(t("batch.dl.csv"), lambda: export.to_csv(index=False).encode("utf-8"),
                               f"{base}_scored.csv", "text/csv", on_click="ignore", key="rep_csv",
                               icon=":material/download:")
    with c2:
        with card(t("rep.batch.xlsx"), t("batch.dl.xlsx_help")):
            st.download_button(t("batch.dl.xlsx"), xlsx, f"{base}_scored.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               on_click="ignore", key="rep_xlsx", icon=":material/table_view:")
    with c3:
        with card(t("rep.batch.calls"), t("rep.batch.calls_desc")):
            k = st.number_input(t("batch.dl.k"), 1, max(len(df), 1), min(500, len(df)), 50, key="rep_k")
            calls = export.nlargest(int(k), "churn_probability")
            st.download_button(t("batch.dl.calls", k=fmt.num(k)), lambda: calls.to_csv(index=False).encode("utf-8"),
                               f"{base}_priority_call_list_top{int(k)}.csv", "text/csv", on_click="ignore",
                               key="rep_calls", icon=":material/call:")

# ---------------------------------------------------------------------------
# Reference files (the original downloads, kept)
# ---------------------------------------------------------------------------
section_header(t("rep.ref.title"), t("rep.ref.sub"))
dictionary = data_dictionary(config, lang)
c1, c2 = st.columns(2)
with c1:
    with card(t("rep.ref.dict"), t("rep.ref.dict_desc")):
        st.download_button(t("rep.ref.dict_dl"), dictionary.to_csv(index=False).encode("utf-8-sig"),
                           f"expresso_data_dictionary_{lang}.csv", "text/csv", key="rep_dict",
                           icon=":material/menu_book:")
    with card(t("rep.ref.meta"), t("rep.ref.meta_desc")):
        st.download_button(t("rep.ref.meta_dl"), json.dumps(meta, indent=2), "model_metadata.json",
                           "application/json", key="rep_meta", icon=":material/data_object:")
with c2:
    with card(t("rep.ref.segments"), t("rep.ref.segments_desc")):
        st.download_button(t("rep.ref.segments_dl"), profiles.to_csv(index=False), "segment_profiles.csv",
                           "text/csv", key="rep_seg", icon=":material/groups:")
    with card(t("rep.ref.features"), t("rep.ref.features_desc", n=len(features))):
        st.download_button(t("rep.ref.features_dl"), "\n".join(features), "model_features.txt", "text/plain",
                           key="rep_feat", icon=":material/list:")

with st.expander(t("rep.ref.dict_preview"), icon=":material/table_view:"):
    st.dataframe(dictionary, hide_index=True, width="stretch")

section_header(t("rep.deploy.title"), t("rep.deploy.sub"))
summary = "\n".join([
    "EXPRESSO CUSTOMER CHURN PREDICTION",
    "",
    f"Model: {meta['model_name']}",
    f"Validation ROC-AUC: {meta['validation_roc_auc']:.4f}",
    f"Training rows: {meta['training_rows']:,}",
    f"Model features: {meta['feature_count']}",
    f"Best iteration: {meta['best_iteration']}",
    f"Python: {meta['python_version']}  LightGBM: {meta['lightgbm_version']}  scikit-learn: "
    f"{meta['scikit_learn_version']}",
    f"Risk thresholds approved: {meta['risk_thresholds_approved']}",
    f"Deployment artifact created: {meta['created_utc']}",
])
st.code(summary, language=None)
st.download_button(t("rep.deploy.dl"), summary, "expresso_deployment_summary.txt", "text/plain", key="rep_deploy",
                   icon=":material/description:")
