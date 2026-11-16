"""Scoring on top of the canonical preprocessing.

Everything here calls ``app.utils.prediction.prepare_customer_features`` and
the bundled LightGBM model unchanged. Large batches are processed in chunks
to keep peak memory low on Streamlit Community Cloud.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.core.features import source_of
from app.core.segments import assign_segments
from app.utils.prediction import prepare_customer_features

DEFAULT_CHUNK = 20_000
TOP_K = 3


def required_columns(config: dict) -> list[str]:
    return [config["id_column"]] + list(config["raw_source_columns"])


def predict_proba(features: pd.DataFrame, model) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def contributions(features: pd.DataFrame, model) -> np.ndarray:
    """LightGBM per-feature contributions in log-odds; last column is the bias."""
    return np.asarray(model.booster_.predict(features, pred_contrib=True))


def group_matrix(feature_columns: list[str], raw_columns: list[str]) -> tuple[list[str], np.ndarray]:
    """(group names, F x G indicator matrix) mapping features to source groups."""
    groups: list[str] = []
    for f in feature_columns:
        g = source_of(f, raw_columns)
        if g not in groups:
            groups.append(g)
    m = np.zeros((len(feature_columns), len(groups)), dtype="float64")
    for i, f in enumerate(feature_columns):
        m[i, groups.index(source_of(f, raw_columns))] = 1.0
    return groups, m


def grouped_contributions(contrib: np.ndarray, feature_columns: list[str],
                          raw_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    groups, m = group_matrix(feature_columns, raw_columns)
    per_group = contrib[:, :-1] @ m
    return pd.DataFrame(per_group, columns=groups), contrib[:, -1]


def risk_deciles(prob: np.ndarray | pd.Series) -> np.ndarray:
    """Decile 1 = riskiest 10% of this batch, 10 = safest."""
    s = pd.Series(np.asarray(prob, dtype="float64"))
    n = len(s)
    if n == 0:
        return np.array([], dtype="int8")
    rank = s.rank(ascending=False, method="first").to_numpy()
    return np.clip(np.ceil(rank / n * 10), 1, 10).astype("int8")


def bands(prob: np.ndarray | pd.Series, low: float, high: float) -> np.ndarray:
    """Exploratory Low / Medium / High keys from user-chosen thresholds."""
    p = np.asarray(prob, dtype="float64")
    return np.where(p >= high, "high", np.where(p >= low, "medium", "low"))


def score_batch(raw_df: pd.DataFrame, bundle: dict, chunk_size: int = DEFAULT_CHUNK, with_reasons: bool = True,
                with_segments: bool = True, top_k: int = TOP_K,
                progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Score a raw batch in chunks.

    Returns one row per input row: id, churn_probability, segment and the
    ``top_k`` source groups that pushed risk up the most (with their
    log-odds contributions).
    """
    config = bundle["preprocessing_config"]
    model = bundle["model"]
    id_col = config["id_column"]
    raw_cols = list(config["raw_source_columns"])
    feat_cols = list(config["feature_columns"])

    if raw_df[id_col].isna().any():
        raise ValueError("Customer IDs contain missing values.")
    if raw_df[id_col].duplicated().any():
        raise ValueError("Customer IDs must be unique within the prediction batch.")

    n = len(raw_df)
    parts = []
    groups, gm = group_matrix(feat_cols, raw_cols)
    for start in range(0, n, chunk_size):
        chunk = raw_df.iloc[start:start + chunk_size]
        feats = prepare_customer_features(chunk, config)
        out = pd.DataFrame({id_col: chunk[id_col].to_numpy(), "churn_probability": predict_proba(feats, model)})
        if with_segments:
            out["segment"] = assign_segments(chunk, feats, bundle)
        if with_reasons:
            contrib = contributions(feats, model)
            per_group = contrib[:, :-1] @ gm
            order = np.argsort(-per_group, axis=1)[:, :top_k]
            top_vals = np.take_along_axis(per_group, order, axis=1)
            for k in range(top_k):
                names = np.array(groups, dtype=object)[order[:, k]]
                out[f"reason_{k + 1}"] = np.where(top_vals[:, k] > 0, names, None)
                out[f"reason_{k + 1}_logodds"] = np.where(top_vals[:, k] > 0, top_vals[:, k], np.nan).astype("float32")
            del contrib, per_group
        parts.append(out)
        del feats
        if progress:
            progress(min((start + len(chunk)) / max(n, 1), 1.0))
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(result):
        result["churn_probability"] = result["churn_probability"].astype("float64")
        result["risk_decile"] = risk_deciles(result["churn_probability"])
    return result


def score_one(raw_row: dict, bundle: dict) -> dict:
    """Probability, segment, grouped and raw contributions for one customer."""
    config = bundle["preprocessing_config"]
    df = pd.DataFrame([raw_row])[required_columns(config)]
    feats = prepare_customer_features(df, config)
    model = bundle["model"]
    prob = float(predict_proba(feats, model)[0])
    contrib = contributions(feats, model)
    grouped, bias = grouped_contributions(contrib, list(config["feature_columns"]), list(config["raw_source_columns"]))
    raw_contrib = pd.Series(contrib[0, :-1], index=config["feature_columns"])
    segment = int(assign_segments(df, feats, bundle)[0])
    return {
        "probability": prob,
        "segment": segment,
        "grouped": grouped.iloc[0].sort_values(key=np.abs, ascending=False),
        "raw": raw_contrib.sort_values(key=np.abs, ascending=False),
        "bias": float(bias[0]),
        "features": feats.iloc[0],
    }


def score_rows(rows: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Probabilities for a small frame of raw rows (used by the What-If sweeps)."""
    config = bundle["preprocessing_config"]
    feats = prepare_customer_features(rows[required_columns(config)], config)
    return predict_proba(feats, bundle["model"])
