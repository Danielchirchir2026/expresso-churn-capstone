"""Labelled data for evaluation: validation artifact, uploads, or a batch."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from app.core import metrics, scoring


class EvalDataError(ValueError):
    """Raised with an i18n code (eval.err.<code>) when a file cannot be used."""

    def __init__(self, code: str, **params):
        super().__init__(code)
        self.code = code
        self.params = params


@dataclass
class EvalData:
    y: np.ndarray
    p: np.ndarray
    source: str  # "validation" | "upload" | "batch"
    name: str
    groups: dict = field(default_factory=dict)  # optional REGION / TENURE arrays


def from_validation(df: pd.DataFrame) -> EvalData:
    groups = {c: df[c].astype("string").fillna("Not recorded").to_numpy() for c in ("REGION", "TENURE") if c in df}
    return EvalData(df["y_true"].to_numpy().astype("int8"), df["y_prob"].to_numpy().astype("float64"),
                    "validation", "validation_predictions.parquet", groups)


def _labels(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    bad = y.notna() & ~y.isin([0, 1])
    if bad.any():
        raise EvalDataError("bad_labels", n=int(bad.sum()))
    return y


def from_frame(df: pd.DataFrame, bundle: dict, name: str, source: str = "upload", progress=None) -> EvalData:
    """Use a labelled file: either already scored (churn_probability / y_prob)
    or raw Expresso columns, which are cleaned and scored here."""
    config = bundle["preprocessing_config"]
    target = config.get("target_column", "CHURN")
    prob_col = next((c for c in ("churn_probability", "y_prob") if c in df.columns), None)
    label_col = target if target in df.columns else ("y_true" if ("y_true" in df.columns and prob_col) else None)
    if label_col is None:
        raise EvalDataError("no_label", col=target)

    work = df.copy()
    work["_y"] = _labels(work[label_col])
    work = work[work["_y"].notna()]
    if prob_col:
        work["_p"] = pd.to_numeric(work[prob_col], errors="coerce")
        work = work[work["_p"].notna()]
    else:
        missing = [c for c in scoring.required_columns(config) if c not in work.columns]
        if missing:
            raise EvalDataError("no_prob_or_raw", cols=", ".join(missing[:6]))
        from app.core import validation

        work = validation.clean(work, config, drop_missing_ids=True, drop_duplicates=True)
        work["_y"] = _labels(work[target])
        scored = scoring.score_batch(work, bundle, with_reasons=False, with_segments=False, progress=progress)
        work["_p"] = scored["churn_probability"].to_numpy()
    y = work["_y"].to_numpy().astype("int8")
    if len(np.unique(y)) < 2:
        raise EvalDataError("one_class")
    groups = {c: work[c].astype("string").fillna("Not recorded").to_numpy() for c in ("REGION", "TENURE")
              if c in work.columns}
    return EvalData(y, work["_p"].to_numpy(dtype="float64"), source, name, groups)


def summary(d: EvalData) -> dict:
    return {
        "n": int(len(d.y)),
        "rate": float(d.y.mean()),
        "auc": float(roc_auc_score(d.y, d.p)),
        "ap": float(average_precision_score(d.y, d.p)),
        "brier": metrics.brier(d.y, d.p),
    }


def roc_points(d: EvalData, max_points: int = 600) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(d.y, d.p, drop_intermediate=True)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.clip(thr, 0, 1)})
    return metrics.downsample(df, max_points)


def pr_points(d: EvalData, max_points: int = 600) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(d.y, d.p)
    df = pd.DataFrame({"precision": prec[:-1], "recall": rec[:-1], "threshold": thr})
    return metrics.downsample(df, max_points)
