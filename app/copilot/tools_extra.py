"""Copilot tools over the exported artifacts, drift, revenue and experiments.

Registered into ``app.copilot.tools.TOOLS`` on import. Every tool returns
aggregates only and says plainly when its source file has not been exported.
"""

from __future__ import annotations

import pandas as pd

from app.copilot.tools import Context, Tool, ToolInputError, _obj, _r, register
from app.core import artifacts, drift, metrics, portfolio
from app.core.features import friendly
from app.core.segments import SEGMENT_IDS, segment_name

DIMENSIONS = ["TENURE", "REGION", "REGULARITY", "MRG", "TOP_PACK", "MONTANT_DECILE", "REVENUE_DECILE",
              "DATA_VOLUME_DECILE", "ON_NET_DECILE"]


def _need(name: str):
    data = artifacts.get_optional(name)
    if data is None:
        raise ToolInputError(f"{artifacts.OPTIONAL_FILES[name]} has not been exported yet "
                             f"(run: {artifacts.EXPORT_COMMAND}). I don't have that data.")
    return data


def eda_breakdown(ctx: Context, column: str) -> dict:
    b = _need("eda_breakdowns")
    if column not in DIMENSIONS:
        raise ToolInputError(f"column must be one of {DIMENSIONS}")
    part = b[b["dimension"] == column].sort_values("order").head(25)
    return {"source": "eda_breakdowns.parquet (training data aggregates)", "dimension": column,
            "levels": [{"level": str(r.level), "customers": int(r.count), "churners": int(r.churners),
                        "churn_rate_pct": _r(r.churn_rate * 100, 2)} for r in part.itertuples()]}


def model_comparison(ctx: Context) -> dict:
    c = _need("model_comparison")
    return {"source": "model_comparison.csv (validation split)",
            "models": [{"model": r.model, "roc_auc": _r(r.roc_auc, 4), "pr_auc": _r(r.pr_auc, 4),
                        "log_loss": _r(r.log_loss, 4), "f1_at_0_5": _r(r.f1_at_0_5, 3),
                        "train_seconds": _r(r.train_seconds, 1), "note": str(r.note)} for r in c.itertuples()]}


def evaluation_at_threshold(ctx: Context, threshold: float) -> dict:
    v = _need("validation_predictions")
    t = float(threshold)
    if not 0 <= t <= 1:
        raise ToolInputError("threshold must be between 0 and 1")
    cm = metrics.confusion_at(v["y_true"], v["y_prob"], t)
    manifest = artifacts.get_optional("export_manifest") or {}
    return {"source": "validation_predictions.parquet", "threshold": t,
            "predictions_from": manifest.get("validation_model", "unknown"),
            "customers": int(len(v)), "flagged": cm["flagged"], "true_churners_caught": cm["tp"],
            "false_alarms": cm["fp"], "missed_churners": cm["fn"],
            "precision_pct": _r(cm["precision"] * 100, 1), "recall_pct": _r(cm["recall"] * 100, 1),
            "f1": _r(cm["f1"], 3), "note": "Exploratory threshold; no risk threshold has been approved."}


def revenue_at_risk(ctx: Context) -> dict:
    p = artifacts.get_segment_profiles().set_index("segment")
    risk = p["customers"] * p["churn_rate"] / 100 * p["median_revenue"]
    out = {"source": "segment_profiles.csv (median-based estimate) + session batch",
           "portfolio_monthly_revenue_at_risk_fcfa": _r(risk.sum(), 0),
           "by_segment": [{"segment": segment_name(s, ctx.lang), "monthly_revenue_at_risk_fcfa": _r(risk[s], 0),
                           "median_monthly_revenue_fcfa": _r(p.loc[s, "median_revenue"], 0)} for s in SEGMENT_IDS],
           "caveat": "Segments with a median revenue of 0 contribute 0 to this estimate."}
    if ctx.batch:
        df = ctx.batch["df"]
        ear = df["churn_probability"] * pd.to_numeric(df["REVENUE"], errors="coerce").fillna(0)
        out["batch"] = {"name": ctx.batch["name"], "synthetic_demo_data": bool(ctx.batch["synthetic"]),
                        "expected_monthly_revenue_at_risk_fcfa": _r(ear.sum(), 0),
                        "top_regions": [{"region": str(k), "fcfa": _r(v, 0)} for k, v in
                                        ear.groupby(df["REGION"].astype("string").fillna("Not recorded")).sum()
                                        .nlargest(5).items()]}
    return out


def drift_report(ctx: Context) -> dict:
    if not ctx.batch:
        raise ToolInputError("No scored batch in this session, so there is nothing to compare with training data.")
    reference = _need("reference_stats")
    rep = ctx.batch.get("drift") or drift.report(ctx.batch["df"], reference, artifacts.get_config())
    rows = sorted(rep["rows"], key=lambda r: -r["psi"])
    return {"source": "reference_stats.json vs session batch", "overall": rep["overall"], "counts": rep["counts"],
            "synthetic_demo_data": bool(ctx.batch["synthetic"]),
            "columns": [{"column": friendly(r["column"], ctx.lang), "psi": _r(r["psi"], 3), "status": r["status"],
                         "missing_training_pct": _r(r["missing_ref"] * 100, 1),
                         "missing_batch_pct": _r(r["missing_new"] * 100, 1), "unseen": r["unseen"][:5]}
                        for r in rows[:8]],
            "thresholds": "green < 0.10, amber 0.10-0.25, red > 0.25"}


def sample_size(ctx: Context, baseline: float | None = None, reduction: float = 20.0, alpha: float = 0.05,
                power: float = 0.8, relative: bool = True) -> dict:
    if baseline is None:
        baseline = portfolio.compute(artifacts.get_segment_profiles(), artifacts.get_metadata()).churn_rate
    p1 = float(baseline) / 100
    p2 = p1 * (1 - float(reduction) / 100) if relative else p1 - float(reduction) / 100
    if not 0 < p2 < p1 < 1:
        raise ToolInputError("Need 0 < churn after the offer < baseline churn < 100%.")
    if alpha not in (0.01, 0.05, 0.1) and not 0 < alpha < 0.5:
        raise ToolInputError("alpha must be between 0 and 0.5")
    n = metrics.sample_size_two_proportions(p1, p2, float(alpha), float(power))
    return {"source": "two-proportion z-test (normal approximation)", "baseline_churn_pct": _r(p1 * 100, 2),
            "target_churn_pct": _r(p2 * 100, 2), "alpha": alpha, "power": power, "customers_per_group": n,
            "customers_total": 2 * n}


for _tool in [
    Tool("eda_breakdown", "Churn rate and counts in the training data by a dimension: " + ", ".join(DIMENSIONS) + ".",
         _obj({"column": {"type": "string", "enum": DIMENSIONS}}, ["column"]), eda_breakdown),
    Tool("model_comparison", "Validation leaderboard: Dummy baseline, Logistic Regression and LightGBM (ROC-AUC, "
         "PR-AUC, log loss, F1 at 0.5, training time).", _obj(), model_comparison),
    Tool("evaluation_at_threshold", "Confusion counts, precision, recall and F1 on the validation split if customers "
         "with probability >= threshold are flagged.",
         _obj({"threshold": {"type": "number", "minimum": 0, "maximum": 1}}, ["threshold"]), evaluation_at_threshold),
    Tool("revenue_at_risk", "Monthly revenue at risk: portfolio estimate by segment and, if a batch is scored, the "
         "expected revenue at risk in it.", _obj(), revenue_at_risk),
    Tool("drift_report", "Data drift (PSI) of the session batch against the training data, per column.", _obj(),
         drift_report),
    Tool("sample_size", "A/B test size for a retention offer: customers per group to detect the reduction. baseline "
         "and reduction are percentages; relative=true means a relative reduction.",
         _obj({"baseline": {"type": "number"}, "reduction": {"type": "number"}, "alpha": {"type": "number"},
               "power": {"type": "number"}, "relative": {"type": "boolean"}}, ["reduction"]), sample_size),
]:
    register(_tool)
