"""Data drift: Population Stability Index (PSI) against the training data.

Bins come from ``reference_stats.json`` (written by the export script):
training deciles plus a "missing" bucket for numeric columns, and the
training categories plus "unseen" and "missing" buckets for categorical
columns. Traffic lights: green < 0.1, amber 0.1-0.25, red > 0.25.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.core import fmt

GREEN, AMBER = 0.10, 0.25
EPS = 1e-4


def psi(expected, actual, eps: float = EPS) -> float:
    """PSI = sum((a - e) * ln(a / e)); proportions are floored at ``eps``."""
    e = np.clip(np.asarray(expected, dtype="float64"), eps, None)
    a = np.clip(np.asarray(actual, dtype="float64"), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "na"
    return "green" if value < GREEN else ("amber" if value <= AMBER else "red")


def numeric_bins(values: pd.Series, ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """(expected, actual) proportions over decile bins + missing."""
    v = pd.to_numeric(values, errors="coerce")
    inner = np.asarray(ref["edges"], dtype="float64")
    present = v.dropna().to_numpy()
    counts = np.bincount(np.searchsorted(inner, present, side="right"), minlength=len(inner) + 1)
    n = max(len(v), 1)
    actual = np.append(counts / n, v.isna().sum() / n)
    expected = np.append(np.asarray(ref["proportions"], dtype="float64"), ref["missing_rate"])
    return expected, actual


def categorical_bins(values: pd.Series, ref: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """(expected, actual, unseen levels) over known levels + unseen + missing."""
    v = values.astype("string").str.strip()
    known = {k: p for k, p in ref["proportions"].items() if k != "__other__"}
    other_ref = ref["proportions"].get("__other__", 0.0)
    n = max(len(v), 1)
    counts = v.value_counts(dropna=True)
    actual = [counts.get(k, 0) / n for k in known]
    expected = list(known.values())
    rest = counts[~counts.index.isin(list(known))]
    unseen = [str(x) for x in rest.index] if "__other__" not in ref["proportions"] else []
    actual.append(rest.sum() / n)
    expected.append(other_ref)
    actual.append(v.isna().sum() / n)
    expected.append(ref["missing_rate"])
    return np.asarray(expected), np.asarray(actual), unseen[:10]


def report(df: pd.DataFrame, reference: dict, config: dict) -> dict:
    """PSI, status and missing-rate change for every column in the reference."""
    rows = []
    for col, ref in reference.get("numeric", {}).items():
        if col not in df.columns:
            continue
        e, a = numeric_bins(df[col], ref)
        value = psi(e, a)
        rows.append({"column": col, "kind": "numeric", "psi": value, "status": status(value),
                     "missing_ref": ref["missing_rate"], "missing_new": float(a[-1]), "unseen": []})
    for col, ref in reference.get("categorical", {}).items():
        if col not in df.columns:
            continue
        e, a, unseen = categorical_bins(df[col], ref)
        value = psi(e, a)
        rows.append({"column": col, "kind": "categorical", "psi": value, "status": status(value),
                     "missing_ref": ref["missing_rate"], "missing_new": float(a[-1]), "unseen": unseen})
    table = pd.DataFrame(rows)
    counts = table["status"].value_counts().to_dict() if len(table) else {}
    overall = "red" if counts.get("red") else ("amber" if counts.get("amber") else "green")
    return {"rows": rows, "counts": {k: int(v) for k, v in counts.items()}, "overall": overall,
            "reference_rows": reference.get("rows"), "batch_rows": int(len(df))}


def distribution_frame(df: pd.DataFrame, reference: dict, col: str, missing_label: str, unseen_label: str) -> pd.DataFrame:
    """Training vs batch proportions per bin, for plotting one column."""
    if col in reference.get("numeric", {}):
        ref = reference["numeric"][col]
        e, a = numeric_bins(df[col], ref)
        edges = [-math.inf] + list(ref["edges"]) + [math.inf]
        labels = [f"{_fmt(edges[i])} - {_fmt(edges[i + 1])}" for i in range(len(edges) - 1)] + [missing_label]
    else:
        ref = reference["categorical"][col]
        e, a, _ = categorical_bins(df[col], ref)
        labels = [k for k in ref["proportions"] if k != "__other__"] + [unseen_label, missing_label]
    return pd.DataFrame({"bin": labels, "training": e, "batch": a})


def _fmt(x: float) -> str:
    if math.isinf(x):
        return "-∞" if x < 0 else "+∞"
    return fmt.num(x, 0) if abs(x) >= 10 else fmt.num(x, 2)
