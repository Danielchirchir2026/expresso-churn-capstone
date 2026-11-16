"""Preset customer profiles built from segment medians.

``segment_profiles.csv`` has medians for revenue, recharges, active days and
data volume only. Those fill the preset; every other usage field is left
empty ("not in profile") unless ``eda_summary.json`` carries per-segment
medians for it (written by scripts/export_artifacts.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.segments import SEGMENT_IDS

NUMERIC = ["MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE", "DATA_VOLUME", "ON_NET", "ORANGE",
           "TIGO", "ZONE1", "ZONE2", "REGULARITY", "FREQ_TOP_PACK"]
FROM_PROFILE = {"REVENUE": "median_revenue", "FREQUENCE_RECH": "median_recharges",
                "REGULARITY": "median_active_days", "DATA_VOLUME": "median_data_volume"}


def preset(seg_id: int, profiles: pd.DataFrame, eda_summary: dict | None = None) -> tuple[dict, set]:
    """Return (raw values, set of columns actually sourced from data)."""
    p = profiles.set_index("segment").loc[int(seg_id)]
    values = {c: np.nan for c in NUMERIC}
    sourced: set = set()

    seg_medians = ((eda_summary or {}).get("segment_medians") or {}).get(str(int(seg_id)))
    if seg_medians:
        for c in NUMERIC:
            v = seg_medians.get(c)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values[c] = float(v)
                sourced.add(c)
        modes = ((eda_summary or {}).get("segment_modes") or {}).get(str(int(seg_id))) or {}
        for c in ("TENURE", "REGION", "TOP_PACK"):
            if modes.get(c):
                values[c] = modes[c]
                sourced.add(c)
        return values, sourced

    # Silent / Inactive is defined by empty usage fields: keep them empty,
    # only the activity count is informative.
    if int(seg_id) == 0:
        values["REGULARITY"] = float(p["median_active_days"])
        return values, {"REGULARITY"}

    for col, prof_col in FROM_PROFILE.items():
        v = float(p[prof_col])
        if col == "DATA_VOLUME" and v == 0 and float(p["data_users"]) < 50:
            continue  # most of the segment records no data usage: leave empty
        values[col] = v
        sourced.add(col)
    return values, sourced


def all_presets(profiles: pd.DataFrame, eda_summary: dict | None = None) -> dict[int, tuple[dict, set]]:
    return {s: preset(s, profiles, eda_summary) for s in SEGMENT_IDS}
