"""Methodology: feature engineering, imbalance, validation, metric, limits and ethics."""

import html

import streamlit as st

from app import navigation
from app.core import artifacts, fmt
from app.core.features import OWN_GROUPS, duplicated_indicators, friendly, kind_key, source_of
from app.i18n import t
from app.ui.components import insight_callout, kpi_row, page_header, section_header

bundle = artifacts.get_bundle()
config = bundle["preprocessing_config"]
model = bundle["model"]
meta = artifacts.get_metadata()
params = model.get_params()
feat_cols = list(config["feature_columns"])
raw_cols = list(config["raw_source_columns"])
eda = artifacts.get_optional("eda_summary")

page_header(t("meth.title"), t("meth.sub"), t("meth.eyebrow"))

# ---------------------------------------------------------------------------
# Feature engineering (built from the config, not by hand)
# ---------------------------------------------------------------------------
section_header(t("meth.fe.title"), t("meth.fe.sub", raw=len(raw_cols), n=len(feat_cols)))
rows = []
for col in raw_cols + list(OWN_GROUPS):
    derived = [f for f in feat_cols if source_of(f, raw_cols) == col]
    if not derived and col not in raw_cols:
        continue
    kinds = sorted({kind_key(f) for f in derived}, key=lambda k: ["raw", "log", "missing", "region_missing",
                                                                   "region_unseen", "onehot", "ordinal", "freq",
                                                                   "missing_count", "rev_per_recharge",
                                                                   "data_share"].index(k))
    rows.append({
        "raw": f"{col} · {friendly(col)}",
        "features": ", ".join(derived) if derived else t("meth.fe.not_used"),
        "n": len(derived),
        "how": "; ".join(t(f"feat_kind.{k}") for k in kinds) if derived else "–",
        "why": " ".join(t(f"meth.why.{k}") for k in kinds) if derived else t("meth.why.unused"),
    })
head = "".join(f"<th>{html.escape(t(f'meth.fe.col_{c}'))}</th>" for c in ("raw", "features", "how", "why"))
body = "".join(
    f"<tr><td><b>{html.escape(r['raw'])}</b><br><span style='opacity:.6;font-size:.8rem'>{r['n']}</span></td>"
    f"<td><code style='font-size:.74rem;word-break:break-word'>{html.escape(r['features'])}</code></td>"
    f"<td>{html.escape(r['how'])}</td><td>{html.escape(r['why'])}</td></tr>" for r in rows)
st.html(f'<table class="hub-table">{"<thead><tr>" + head + "</tr></thead>"}<tbody>{body}</tbody></table>')
st.caption(t("meth.fe.caption", n=len(feat_cols)))
c1, c2 = st.columns(2)
with c1:
    navigation.link("model", label=t("meth.fe.to_model"))
with c2:
    navigation.link("reports", label=t("meth.fe.to_dict"))

# ---------------------------------------------------------------------------
# Imbalance, validation, metric
# ---------------------------------------------------------------------------
section_header(t("meth.train.title"), t("meth.train.sub"))
weighted = bool(params.get("class_weight")) or bool(params.get("is_unbalance")) or \
    (params.get("scale_pos_weight") not in (None, 1, 1.0))
churn_rate = (eda or {}).get("churn_rate")
kpi_row([
    dict(label=t("meth.train.rate"), value=fmt.pct(churn_rate * 100, 2) if churn_rate is not None else "–",
         foot=t("meth.train.rate_foot") if churn_rate is not None else t("meth.train.rate_missing")),
    dict(label=t("meth.train.weighting"), value=t("meth.train.weighted") if weighted else t("meth.train.unweighted"),
         foot=f"class_weight = {params.get('class_weight')}"),
    dict(label=t("meth.train.seed"), value=str(meta["validation_split_seed"]), foot=t("meth.train.seed_foot")),
    dict(label=t("meth.train.trees"), value=f"{model.booster_.num_trees()} / {meta['best_iteration']}",
         foot=t("meth.train.trees_foot", ratio=fmt.num(model.booster_.num_trees() / meta["best_iteration"], 2))),
], cols=4)
insight_callout(t("meth.train.imbalance_weighted") if weighted else t("meth.train.imbalance"), "info",
                tag=t("meth.train.imbalance_tag"))
insight_callout(t("meth.train.validation", seed=meta["validation_split_seed"], best=meta["best_iteration"],
                  trees=model.booster_.num_trees(), lr=params.get("learning_rate"), leaves=params.get("num_leaves")),
                "info", tag=t("meth.train.validation_tag"))
insight_callout(t("meth.train.metric", auc=fmt.num(meta["validation_roc_auc"], 4)), "info",
                tag=t("meth.train.metric_tag"))

# ---------------------------------------------------------------------------
# Limitations and ethics
# ---------------------------------------------------------------------------
section_header(t("meth.ethics.title"), t("meth.ethics.sub"))
region_missing = (eda or {}).get("missing_pct", {}).get("REGION")
items = [
    ("warn", t("meth.ethics.duplicates", n=len(duplicated_indicators(feat_cols)))),
    ("warn", t("meth.ethics.thresholds")),
    ("info", t("meth.ethics.snapshot")),
    ("warn", t("meth.ethics.region", pct=fmt.pct(region_missing, 0)) if region_missing is not None
     else t("meth.ethics.region_nopct")),
    ("info", t("meth.ethics.decision_support")),
    ("info", t("meth.ethics.privacy")),
]
for tone, text in items:
    insight_callout(text, tone)
navigation.link("evaluation", label=t("meth.ethics.fairness_link"))
