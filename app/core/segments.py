"""Single source of truth for customer segments.

Segment ids are the 0-4 index used in ``artifacts/segment_profiles.csv``.
The mapping below was verified two ways against the deployment bundle:

* ``segmentation_config["segment_relabel"] == {0: 4, 1: 1, 2: 3, 3: 2}`` and
  ``silent_segment == 0``, and
* the KMeans cluster centres: cluster 1 (highest revenue, ~53 active days)
  -> segment 1, cluster 3 (highest data volume) -> segment 2, cluster 2
  (~26 active days, highest off-net share) -> segment 3, cluster 0
  (lowest activity) -> segment 4. These match the profile medians.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.i18n import t, tl

SEGMENT_KEYS = {
    0: "silent",
    1: "high_value",
    2: "data_engaged",
    3: "light_active",
    4: "low_engagement",
}
SEGMENT_IDS = list(SEGMENT_KEYS)

# Categorical palette (validated with the dataviz validator against the
# app's navy and light surfaces). Fixed per segment id, never by rank.
SEGMENT_COLORS = {
    "dark": {0: "#3987e5", 1: "#d95926", 2: "#199e70", 3: "#c98500", 4: "#d55181"},
    "light": {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#eda100", 4: "#e87ba4"},
}


def segment_name(seg_id, lang: str | None = None) -> str:
    return t(f"seg.{SEGMENT_KEYS[int(seg_id)]}.name", lang)


def segment_blurb(seg_id, lang: str | None = None) -> str:
    return t(f"seg.{SEGMENT_KEYS[int(seg_id)]}.blurb", lang)


def segment_actions(seg_id, lang: str | None = None) -> list[str]:
    return tl(f"seg.{SEGMENT_KEYS[int(seg_id)]}.actions", lang)


def with_names(profiles: pd.DataFrame, lang: str | None = None) -> pd.DataFrame:
    out = profiles.copy()
    out["segment"] = out["segment"].astype(int)
    out["segment_name"] = out["segment"].map(lambda s: segment_name(s, lang))
    return out


# ---------------------------------------------------------------------------
# Batch segment assignment (reconstructed from the bundle's segmentation
# model). The KMeans model, scaler and relabel map come from the bundle;
# the feature recipe (log1p of usage with missing -> 0, off-net share of
# calls, REGULARITY) and the silent rule (MISSING_COUNT >= max_missing_count)
# are reconstructed from ``segmentation_config``. The export script checks
# the reconstruction against segment_profiles.csv on the training data.
# ---------------------------------------------------------------------------
def segmentation_features(raw_df: pd.DataFrame, seg_config: dict) -> pd.DataFrame:
    usage = raw_df[seg_config["usage_columns"]].apply(pd.to_numeric, errors="coerce")
    usage = usage.clip(lower=0).fillna(0.0)
    feats = pd.DataFrame(index=raw_df.index)
    for column in seg_config["usage_columns"]:
        feats[f"LOG_{column}"] = np.log1p(usage[column])
    calls = usage["ON_NET"] + usage["ORANGE"] + usage["TIGO"]
    offnet = (usage["ORANGE"] + usage["TIGO"]) / calls.replace(0, np.nan)
    feats["OFFNET_SHARE"] = offnet.fillna(0.0)
    feats["REGULARITY"] = pd.to_numeric(raw_df["REGULARITY"], errors="coerce").fillna(0.0)
    return feats[seg_config["cluster_features"]]


def assign_segments(raw_df: pd.DataFrame, model_features: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Return a segment id (0-4) for each row."""
    seg_config = bundle["segmentation_config"]
    feats = segmentation_features(raw_df, seg_config)
    scaled = bundle["segmentation_scaler"].transform(feats.astype("float64"))
    clusters = bundle["segmentation_model"].predict(scaled)
    relabel = seg_config["segment_relabel"]
    segments = np.array([relabel[int(c)] for c in clusters], dtype="int8")
    silent = model_features["MISSING_COUNT"].to_numpy() >= seg_config["max_missing_count"]
    segments[silent] = seg_config["silent_segment"]
    return segments
