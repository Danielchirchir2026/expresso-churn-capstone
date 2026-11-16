"""Data dictionary built from the preprocessing config (not maintained by hand)."""

from __future__ import annotations

import pandas as pd

from app.core.features import OWN_GROUPS, describe, friendly, source_of, transformation
from app.i18n import t


def data_dictionary(config: dict, lang: str = "en") -> pd.DataFrame:
    raw = list(config["raw_source_columns"])
    feats = list(config["feature_columns"])
    missable = set(config["columns_with_missing_values"])
    categorical = set(config["categorical_columns"])
    id_col, target = config["id_column"], config.get("target_column", "CHURN")
    allowed = {
        "TENURE": ", ".join(config["tenure_order"]),
        "REGION": ", ".join(r for r in config["region_values"] if r != "MISSING"),
        "MRG": "NO",
        "TOP_PACK": t("dict.packs", lang, n=len(config["top_pack_frequency"]) - 2),
        "REGULARITY": "0-90",
    }
    rows = []
    for col in [id_col] + raw + [target]:
        if col == id_col:
            kind = t("dict.kind.id", lang)
        elif col == target:
            kind = t("dict.kind.label", lang)
        else:
            kind = t("dict.kind.categorical", lang) if col in categorical else t("dict.kind.numeric", lang)
        derived = [f for f in feats if source_of(f, raw) == col]
        rows.append({
            "column": col,
            "name": friendly(col, lang),
            "description": describe(col, lang),
            "type": kind,
            "can_be_empty": t("dict.yes", lang) if col in missable else t("dict.no", lang),
            "used_by_model": t("dict.yes", lang) if derived else t("dict.no", lang),
            "allowed_values": allowed.get(col, ""),
            "engineered_features": "; ".join(f"{f} ({transformation(f, lang)})" for f in derived),
        })
    for group in OWN_GROUPS:
        rows.append({
            "column": group, "name": friendly(group, lang), "description": describe(group, lang),
            "type": t("dict.kind.engineered", lang), "can_be_empty": t("dict.yes", lang) if group != "MISSING_COUNT"
            else t("dict.no", lang), "used_by_model": t("dict.yes", lang), "allowed_values": "",
            "engineered_features": f"{group} ({transformation(group, lang)})",
        })
    return pd.DataFrame(rows)
