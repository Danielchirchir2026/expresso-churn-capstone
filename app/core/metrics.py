"""Pure evaluation and business maths (no Streamlit), unit-tested by hand.

``outcomes`` can be true labels (0/1) or predicted probabilities; with
probabilities the curves describe *expected* churners, which is what the
Retention Planner uses when no labels exist.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gains and lift
# ---------------------------------------------------------------------------
def _order(scores) -> np.ndarray:
    # Stable sort, highest score first; ties keep input order.
    return np.argsort(-np.asarray(scores, dtype="float64"), kind="mergesort")


def gains_curve(scores, outcomes) -> tuple[np.ndarray, np.ndarray]:
    """(fraction targeted, fraction of outcomes captured), both starting at 0."""
    s = np.asarray(scores, dtype="float64")
    o = np.asarray(outcomes, dtype="float64")[_order(s)]
    n, total = len(o), o.sum()
    x = np.arange(0, n + 1) / max(n, 1)
    y = np.concatenate([[0.0], np.cumsum(o) / total]) if total > 0 else np.zeros(n + 1)
    return x, y


def captured_at(scores, outcomes, frac: float) -> float:
    """Share of all outcomes captured by targeting the top ``frac`` of customers."""
    s = np.asarray(scores, dtype="float64")
    o = np.asarray(outcomes, dtype="float64")[_order(s)]
    k = int(math.ceil(frac * len(o)))
    total = o.sum()
    return float(o[:k].sum() / total) if total > 0 else 0.0


def lift_at(scores, outcomes, frac: float) -> float:
    return captured_at(scores, outcomes, frac) / frac if frac > 0 else float("nan")


def gains_table(scores, outcomes, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: customers, outcomes, rate, cumulative capture and lift."""
    s = np.asarray(scores, dtype="float64")
    o = np.asarray(outcomes, dtype="float64")[_order(s)]
    n = len(o)
    edges = np.linspace(0, n, n_bins + 1).round().astype(int)
    rows, cum = [], 0.0
    total = o.sum()
    base_rate = total / n if n else float("nan")
    for i in range(n_bins):
        part = o[edges[i]:edges[i + 1]]
        cum += part.sum()
        rate = part.mean() if len(part) else float("nan")
        rows.append({
            "bin": i + 1,
            "customers": int(len(part)),
            "outcomes": float(part.sum()),
            "rate": float(rate),
            "cum_captured": float(cum / total) if total else float("nan"),
            "lift": float(rate / base_rate) if base_rate else float("nan"),
            "cum_lift": float((cum / total) / (edges[i + 1] / n)) if total and n else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Retention economics
# ---------------------------------------------------------------------------
def retention_plan(prob, value, offer_cost: float, save_rate: float, rank_by: str = "prob") -> tuple[pd.DataFrame, np.ndarray]:
    """Cumulative economics of contacting the top-k customers, k = 1..N.

    Expected churners reached  = sum of p over the contacted customers
    Expected customers saved   = churners reached x save rate
    Expected revenue saved     = sum of p x value x save rate
    Campaign cost              = k x offer cost (every contacted customer)
    Net                        = revenue saved - cost
    """
    p = np.asarray(prob, dtype="float64")
    v = np.nan_to_num(np.asarray(value, dtype="float64"), nan=0.0)
    key = p * v if rank_by == "value" else p
    order = _order(key)
    p, v = p[order], v[order]
    k = np.arange(1, len(p) + 1)
    reached = np.cumsum(p)
    saved_rev = np.cumsum(p * v) * save_rate
    cost = k * float(offer_cost)
    plan = pd.DataFrame({"k": k, "reached": reached, "saved": reached * save_rate, "revenue_saved": saved_rev,
                         "cost": cost, "net": saved_rev - cost})
    return plan, order


def best_k(plan: pd.DataFrame, capacity: int) -> int:
    """k (<= capacity) with the highest net value; 0 when nothing is profitable."""
    if plan.empty:
        return 0
    capped = plan.iloc[: max(int(capacity), 0)]
    if capped.empty or capped["net"].max() <= 0:
        return 0
    return int(capped.loc[capped["net"].idxmax(), "k"])


def downsample(df: pd.DataFrame, max_points: int = 600) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, max_points).round().astype(int))
    return df.iloc[idx]


# ---------------------------------------------------------------------------
# Classification at a threshold, threshold sweeps, calibration
# ---------------------------------------------------------------------------
def confusion_at(y_true, prob, threshold: float) -> dict:
    """Counts and rates when customers with prob >= threshold are flagged."""
    y = np.asarray(y_true).astype(bool)
    flag = np.asarray(prob, dtype="float64") >= threshold
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    tn = int((~flag & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp and (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "accuracy": (tp + tn) / max(len(y), 1), "flagged": tp + fp}


def threshold_sweep(y_true, prob, grid=None) -> pd.DataFrame:
    """tp/fp/fn/tn, precision, recall and F1 for every threshold in ``grid``.

    Uses one sort and cumulative sums, so it is fast on hundreds of
    thousands of rows.
    """
    y = np.asarray(y_true).astype("int64")
    p = np.asarray(prob, dtype="float64")
    grid = np.linspace(0, 1, 201) if grid is None else np.asarray(grid, dtype="float64")
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    pos_total, n = int(y.sum()), len(y)
    # number of rows with p < t, and positives among them
    idx = np.searchsorted(p_sorted, grid, side="left")
    pos_below = np.concatenate([[0], np.cumsum(y_sorted)])[idx]
    flagged = n - idx
    tp = pos_total - pos_below
    fp = flagged - tp
    fn = pos_total - tp
    tn = n - flagged - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, tp / np.maximum(flagged, 1), np.nan)
        recall = tp / max(pos_total, 1)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({"threshold": grid, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged": flagged,
                         "precision": precision, "recall": recall, "f1": np.nan_to_num(f1)})


def threshold_cost(y_true, prob, cost_fp: float, cost_fn: float, grid=None) -> pd.DataFrame:
    """Total cost = false alarms x cost_fp + missed churners x cost_fn, per threshold."""
    sweep = threshold_sweep(y_true, prob, grid)
    sweep["cost"] = sweep["fp"] * float(cost_fp) + sweep["fn"] * float(cost_fn)
    return sweep


def best_cost_threshold(cost_table: pd.DataFrame) -> float:
    """Threshold with the lowest total cost (lowest threshold wins ties)."""
    return float(cost_table.loc[cost_table["cost"].idxmin(), "threshold"])


def calibration_table(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    """Observed churn rate vs mean predicted probability in equal-count bins."""
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(prob, dtype="float64")
    order = np.argsort(p, kind="mergesort")
    rows = []
    for part in np.array_split(order, n_bins):
        if len(part):
            rows.append({"mean_pred": float(p[part].mean()), "observed": float(y[part].mean()), "n": int(len(part))})
    return pd.DataFrame(rows)


def brier(y_true, prob) -> float:
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(prob, dtype="float64")
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Fairness: performance by group
# ---------------------------------------------------------------------------
def group_metrics(y_true, prob, groups, threshold: float, min_size: int = 30, auc_gap: float = 0.05) -> pd.DataFrame:
    """AUC, recall, precision, mean probability and observed churn per group.

    A group is flagged when its AUC is more than ``auc_gap`` below the
    overall AUC. Groups that are too small or contain one class only get
    no AUC (and are never flagged).
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype="float64")
    g = pd.Series(np.asarray(groups, dtype=object)).fillna("Not recorded").astype(str).to_numpy()
    overall = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    rows = []
    for level in pd.unique(g):
        m = g == level
        yy, pp_ = y[m], p[m]
        n = int(m.sum())
        enough = n >= min_size and len(np.unique(yy)) == 2
        auc = float(roc_auc_score(yy, pp_)) if enough else float("nan")
        flag = pp_ >= threshold
        tp = int((flag & (yy == 1)).sum())
        rows.append({
            "group": level, "n": n, "churn_rate": float(yy.mean()), "mean_pred": float(pp_.mean()),
            "auc": auc, "recall": tp / max(int(yy.sum()), 1) if yy.sum() else float("nan"),
            "precision": tp / int(flag.sum()) if flag.sum() else float("nan"),
            "auc_gap": auc - overall if enough else float("nan"),
            "flagged": bool(enough and auc < overall - auc_gap),
            "too_small": not enough,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
