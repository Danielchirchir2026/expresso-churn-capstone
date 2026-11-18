"""Map engineered model features back to the raw Expresso columns.

Grouping is derived from the feature names and the preprocessing config,
not maintained by hand: ``LOG1P_MONTANT``, ``MONTANT``, ``MONTANT_MISSING``
and ``is_missing_MONTANT`` all belong to the source group ``MONTANT``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.i18n import t

# Derived ratios / counts form their own groups.
OWN_GROUPS = ("MISSING_COUNT", "REVENUE_PER_RECHARGE", "DATA_SHARE")

# Columns customers can act on (the what-if levers on the Customer Journey page)
ACTIONABLE = ["REGULARITY", "FREQUENCE_RECH", "MONTANT", "REVENUE", "ARPU_SEGMENT", "FREQUENCE",
              "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "FREQ_TOP_PACK"]


def source_of(feature: str, raw_columns: list[str]) -> str:
    """Return the raw source column (or derived group) of a model feature."""
    if feature in OWN_GROUPS or feature in raw_columns:
        return feature
    if feature.startswith("LOG1P_"):
        return feature[len("LOG1P_"):]
    if feature.startswith("is_missing_"):
        return feature[len("is_missing_"):]
    if feature == "TENURE_ORDINAL":
        return "TENURE"
    if feature == "TOP_PACK_FREQ":
        return "TOP_PACK"
    if feature.endswith("_MISSING") and feature[: -len("_MISSING")] in raw_columns:
        return feature[: -len("_MISSING")]
    if feature.startswith("REGION_"):
        return "REGION"
    return feature


def feature_groups(feature_columns: list[str], raw_columns: list[str]) -> dict[str, str]:
    return {f: source_of(f, raw_columns) for f in feature_columns}


def friendly(group: str, lang: str | None = None) -> str:
    """Plain-English name of a source column or derived group."""
    key = f"feat.{group}.name"
    value = t(key, lang)
    return group if value == key else value


def describe(group: str, lang: str | None = None) -> str:
    key = f"feat.{group}.desc"
    value = t(key, lang)
    return "" if value == key else value


def kind_key(feature: str) -> str:
    """Which transformation built a model feature (i18n suffix under feat_kind.*)."""
    if feature.startswith("LOG1P_"):
        return "log"
    if feature == "REGION_MISSING":
        return "region_missing"
    if feature == "REGION_Missing":
        return "region_unseen"
    if feature.startswith("is_missing_") or (feature.endswith("_MISSING") and not feature.startswith("REGION_")):
        return "missing"
    if feature.startswith("REGION_"):
        return "onehot"
    return {"TENURE_ORDINAL": "ordinal", "TOP_PACK_FREQ": "freq", "MISSING_COUNT": "missing_count",
            "REVENUE_PER_RECHARGE": "rev_per_recharge", "DATA_SHARE": "data_share"}.get(feature, "raw")


def transformation(feature: str, lang: str | None = None) -> str:
    """Short description of how one engineered feature is built."""
    return t(f"feat_kind.{kind_key(feature)}", lang)


@st.cache_data(show_spinner=False)
def global_importance(_model, feature_columns: tuple, raw_columns: tuple) -> pd.DataFrame:
    """Gain and split importance per feature, with source group and shares."""
    booster = _model.booster_
    gain = booster.feature_importance("gain")
    split = booster.feature_importance("split")
    df = pd.DataFrame({"feature": booster.feature_name(), "gain": gain, "split": split})
    df["group"] = [source_of(f, list(raw_columns)) for f in df["feature"]]
    df["gain_share"] = df["gain"] / df["gain"].sum() * 100
    df["split_share"] = df["split"] / df["split"].sum() * 100
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


def grouped_importance(imp: pd.DataFrame) -> pd.DataFrame:
    out = (imp.groupby("group", as_index=False)[["gain", "split", "gain_share", "split_share"]].sum()
           .sort_values("gain", ascending=False).reset_index(drop=True))
    out["n_features"] = out["group"].map(imp.groupby("group").size())
    return out


def duplicated_indicators(feature_columns: list[str]) -> list[tuple[str, str]]:
    """(X_MISSING, is_missing_X) pairs that encode the same information."""
    pairs = []
    for f in feature_columns:
        if f.startswith("is_missing_"):
            base = f[len("is_missing_"):]
            if f"{base}_MISSING" in feature_columns:
                pairs.append((f"{base}_MISSING", f))
    return pairs


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")
