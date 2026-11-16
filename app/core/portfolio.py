"""Portfolio KPIs and auto-generated insights, all computed from
``segment_profiles.csv`` and ``model_metadata.json`` (nothing hard-coded)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.core import fmt
from app.core.segments import segment_name
from app.i18n import t


@dataclass(frozen=True)
class Portfolio:
    total_customers: int
    est_churners: float
    churn_rate: float  # percent
    roc_auc: float
    revenue_concentration: float  # High-Value Active share of revenue (%)
    churners_from_silent: float  # Silent / Inactive share of churners (%)
    active_days_benchmark: float
    data_users_benchmark: float
    offnet_benchmark: float
    recharges_benchmark: float


def compute(profiles: pd.DataFrame, metadata: dict) -> Portfolio:
    p = profiles.set_index("segment")
    total = float(p["customers"].sum())
    churners = float((p["customers"] * p["churn_rate"] / 100).sum())
    w = p["customers"] / total
    return Portfolio(
        total_customers=int(total),
        est_churners=churners,
        churn_rate=churners / total * 100,
        roc_auc=float(metadata["validation_roc_auc"]),
        revenue_concentration=float(p.loc[1, "pct_of_revenue"]),
        churners_from_silent=float(p.loc[0, "pct_of_churners"]),
        active_days_benchmark=float((w * p["median_active_days"]).sum()),
        data_users_benchmark=float((w * p["data_users"]).sum()),
        offnet_benchmark=float((w * p["offnet_share"]).sum()),
        recharges_benchmark=float((w * p["median_recharges"]).sum()),
    )


def executive_insights(profiles: pd.DataFrame, pf: Portfolio) -> list[tuple[str, str]]:
    """Return 3-5 (tone, text) insights computed from the profiles."""
    p = profiles.set_index("segment")
    out: list[tuple[str, str]] = []

    worst = p["churn_rate"].idxmax()
    out.append(("risk", t(
        "insights.exec.highest_churn",
        segment=segment_name(worst),
        rate=fmt.pct(p.loc[worst, "churn_rate"], 1),
        multiple=fmt.num(p.loc[worst, "churn_rate"] / pf.churn_rate, 1),
        share=fmt.pct(p.loc[worst, "pct_of_churners"], 0),
    )))

    top_rev = p["pct_of_revenue"].idxmax()
    out.append(("good", t(
        "insights.exec.revenue_concentration",
        segment=segment_name(top_rev),
        rev=fmt.pct(p.loc[top_rev, "pct_of_revenue"], 0),
        cust=fmt.pct(p.loc[top_rev, "pct_customers"], 0),
        rate=fmt.pct(p.loc[top_rev, "churn_rate"], 2),
    )))

    low_value = p[p["pct_of_revenue"] < 2.0]
    if not low_value.empty:
        out.append(("warn", t(
            "insights.exec.value_mismatch",
            churners=fmt.pct(low_value["pct_of_churners"].sum(), 0),
            revenue=fmt.pct(low_value["pct_of_revenue"].sum(), 1),
            n=len(low_value),
        )))

    exposure = p["pct_of_revenue"] * p["churn_rate"] / 100
    protect = exposure.idxmax()
    out.append(("info", t(
        "insights.exec.value_exposure",
        segment=segment_name(protect),
        rev=fmt.pct(p.loc[protect, "pct_of_revenue"], 1),
        rate=fmt.pct(p.loc[protect, "churn_rate"], 1),
    )))

    best = p["churn_rate"].idxmin()
    gap = p.loc[best, "median_active_days"] - p.loc[worst, "median_active_days"]
    if gap > 0:
        out.append(("info", t(
            "insights.exec.activity_gap",
            best=segment_name(best),
            best_days=fmt.num(p.loc[best, "median_active_days"]),
            worst=segment_name(worst),
            worst_days=fmt.num(p.loc[worst, "median_active_days"]),
        )))
    return out[:5]
