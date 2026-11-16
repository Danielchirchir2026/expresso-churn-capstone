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
