"""Cached access to every artifact the app reads.

Required artifacts ship with the repo. Optional artifacts are produced by
``scripts/export_artifacts.py``; every loader returns ``None`` when the
file is absent so pages can show an empty state instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.utils.prediction import BUNDLE_PATH, PROJECT_DIR, load_bundle

ARTIFACT_DIR = PROJECT_DIR / "artifacts"
EXPORT_COMMAND = "python scripts/export_artifacts.py --train path/to/Train.csv"

OPTIONAL_FILES = {
    "eda_summary": "eda_summary.json",
    "eda_breakdowns": "eda_breakdowns.parquet",
    "validation_predictions": "validation_predictions.parquet",
    "model_comparison": "model_comparison.csv",
    "reference_stats": "reference_stats.json",
    "export_manifest": "export_manifest.json",
}


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the churn model…")
def get_bundle() -> dict:
    return load_bundle()


def get_model():
    return get_bundle()["model"]


def get_config() -> dict:
    return get_bundle()["preprocessing_config"]


def model_status() -> tuple[bool, str]:
    """(loaded?, message) for the sidebar badge. Never raises."""
    try:
        bundle = get_bundle()
        n = bundle["model"].n_features_in_
        return True, f"{n} features"
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Required small artifacts
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    with open(ARTIFACT_DIR / "model_metadata.json", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def get_segment_profiles() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "segment_profiles.csv")


@st.cache_data(show_spinner=False)
def get_feature_columns() -> list[str]:
    with open(ARTIFACT_DIR / "feature_columns.json", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Optional exported artifacts
# ---------------------------------------------------------------------------
def optional_path(name: str) -> Path:
    return ARTIFACT_DIR / OPTIONAL_FILES[name]


def has_artifact(name: str) -> bool:
    return optional_path(name).exists()


def _read(path: Path):
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported artifact type: {path.suffix}")


@st.cache_resource(show_spinner=False)
def _load_optional(name: str, mtime: float):
    """Read-only shared copy (cache_resource avoids copying large frames)."""
    return _read(optional_path(name))


def get_optional(name: str):
    """Return the parsed optional artifact, or ``None`` if not exported yet."""
    path = optional_path(name)
    if not path.exists():
        return None
    try:
        return _load_optional(name, path.stat().st_mtime)
    except Exception:
        return None


def bundle_size_mb() -> float:
    return BUNDLE_PATH.stat().st_size / 1e6
