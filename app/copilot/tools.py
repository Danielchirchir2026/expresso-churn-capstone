"""Copilot tools: safe, aggregate-only views of the app's data.

Every tool returns a small JSON-serialisable dict. No tool returns raw
customer rows in bulk: ``top_customers`` is capped at 20 IDs with their
probability, segment and reason labels only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from app.core import artifacts, portfolio, presets, scoring
from app.core.features import ACTIONABLE, friendly, global_importance, grouped_importance
from app.core.segments import SEGMENT_IDS, SEGMENT_KEYS, segment_actions, segment_name, with_names

MAX_TOP = 20
RISK_BY_COLUMNS = ["REGION", "TENURE", "segment", "risk_decile", "TOP_PACK"]


class ToolInputError(ValueError):
    pass


@dataclass
class Context:
    """What the tools may read. Built fresh for each question."""
    batch: dict | None = None
    last_customer: dict | None = None
    thresholds: tuple | None = None
    lang: str = "en"
    extra: dict = field(default_factory=dict)


def _r(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else round(v, nd)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------
def portfolio_overview(ctx: Context) -> dict:
    profiles = with_names(artifacts.get_segment_profiles(), ctx.lang)
    pf = portfolio.compute(profiles, artifacts.get_metadata())
    return {
        "source": "segment_profiles.csv + model_metadata.json",
        "total_customers": pf.total_customers,
        "overall_churn_rate_pct": _r(pf.churn_rate, 2),
        "estimated_churners": round(pf.est_churners),
        "validation_roc_auc": _r(pf.roc_auc, 4),
        "high_value_active_revenue_share_pct": _r(pf.revenue_concentration, 2),
        "silent_inactive_share_of_churners_pct": _r(pf.churners_from_silent, 2),
        "segments": [
            {"segment": r["segment_name"], "customers": int(r["customers"]),
             "pct_customers": _r(r["pct_customers"], 2), "churn_rate_pct": _r(r["churn_rate"], 2),
             "pct_of_churners": _r(r["pct_of_churners"], 2), "pct_of_revenue": _r(r["pct_of_revenue"], 2)}
            for _, r in profiles.iterrows()
        ],
        "insights": [text for _, text in portfolio.executive_insights(profiles, pf)],
    }


def segment_profile(ctx: Context, name: str) -> dict:
    profiles = with_names(artifacts.get_segment_profiles(), ctx.lang).set_index("segment")
    key = str(name).strip().lower()
    match = None
    for sid in SEGMENT_IDS:
        candidates = {str(sid), SEGMENT_KEYS[sid], segment_name(sid, "en").lower(), segment_name(sid, "fr").lower()}
        if key in candidates or any(key and key in c for c in candidates):
            match = sid
            break
    if match is None:
        raise ToolInputError(f"Unknown segment '{name}'. Use one of: "
                             + ", ".join(segment_name(s, "en") for s in SEGMENT_IDS))
    r = profiles.loc[match]
    pf = portfolio.compute(profiles.reset_index(), artifacts.get_metadata())
    return {
        "source": "segment_profiles.csv",
        "segment": segment_name(match, ctx.lang),
        "customers": int(r["customers"]),
        "pct_customers": _r(r["pct_customers"], 2),
        "churn_rate_pct": _r(r["churn_rate"], 2),
        "portfolio_churn_rate_pct": _r(pf.churn_rate, 2),
        "pct_of_churners": _r(r["pct_of_churners"], 2),
        "pct_of_revenue": _r(r["pct_of_revenue"], 2),
        "median_monthly_revenue_fcfa": _r(r["median_revenue"], 0),
        "median_recharges": _r(r["median_recharges"], 1),
        "median_active_days_of_90": _r(r["median_active_days"], 1),
        "data_users_pct": _r(r["data_users"], 2),
        "offnet_call_share_pct": _r(r["offnet_share"], 2),
        "recommended_actions": segment_actions(match, ctx.lang),
    }


def model_card(ctx: Context) -> dict:
    meta = artifacts.get_metadata()
    model = artifacts.get_model()
    config = artifacts.get_config()
    imp = grouped_importance(global_importance(model, tuple(config["feature_columns"]),
                                               tuple(config["raw_source_columns"])))
    return {
        "source": "model_metadata.json + model bundle",
        "model": meta["model_name"],
        "validation_roc_auc": _r(meta["validation_roc_auc"], 4),
        "validation_split_seed": meta["validation_split_seed"],
        "training_rows": meta["training_rows"],
        "feature_count": meta["feature_count"],
        "trees_in_deployed_model": int(model.booster_.num_trees()),
        "best_validation_iteration": meta["best_iteration"],
        "risk_thresholds_approved": bool(meta["risk_thresholds_approved"]),
        "top_drivers_by_gain": [{"driver": friendly(g, ctx.lang), "gain_share_pct": _r(s, 2)}
                                for g, s in zip(imp["group"].head(8), imp["gain_share"].head(8))],
        "limitations": [
            "Duplicated missing-value indicator features (X_MISSING and is_missing_X).",
            "Deployed model appears refit on all rows incl. the validation split (more trees than best iteration).",
            "'Region not recorded' is the strongest signal; it proxies inactivity, not geography.",
            "90-day snapshot; no business-approved risk thresholds.",
        ],
    }


def _batch_df(ctx: Context) -> pd.DataFrame:
    if not ctx.batch:
        raise ToolInputError("No scored batch in this session. Ask the user to score one on Batch Scoring.")
    return ctx.batch["df"]


def batch_summary(ctx: Context) -> dict:
    df = _batch_df(ctx)
    p = df["churn_probability"]
    out = {
        "source": f"session batch '{ctx.batch['name']}'",
        "synthetic_demo_data": bool(ctx.batch["synthetic"]),
        "customers": int(len(df)),
        "mean_probability_pct": _r(p.mean() * 100, 2),
        "median_probability_pct": _r(p.median() * 100, 2),
        "expected_churners": _r(p.sum(), 1),
        "top_decile_mean_probability_pct": _r(p[df["risk_decile"] == 1].mean() * 100, 2),
        "by_segment": [
            {"segment": segment_name(s, ctx.lang), "customers": int(g.size),
             "mean_probability_pct": _r(g.mean() * 100, 2)}
            for s, g in df.groupby("segment")["churn_probability"]
        ],
    }
    if ctx.thresholds:
        low, high = ctx.thresholds
        out["exploratory_bands_not_approved"] = {
            "medium_from": low, "high_from": high,
            "high_count": int((p >= high).sum()), "medium_count": int(((p >= low) & (p < high)).sum()),
        }
    if "REVENUE" in df:
        rev = pd.to_numeric(df["REVENUE"], errors="coerce").fillna(0)
        out["expected_monthly_revenue_at_risk_fcfa"] = _r((p * rev).sum(), 0)
    return out


def top_customers(ctx: Context, n: int = 10) -> dict:
    df = _batch_df(ctx)
    n = int(n)
    if not 1 <= n <= MAX_TOP:
        raise ToolInputError(f"n must be between 1 and {MAX_TOP}.")
    id_col = artifacts.get_config()["id_column"]
    top = df.nlargest(n, "churn_probability")
    rows = []
    for _, r in top.iterrows():
        reasons = [friendly(r[c], ctx.lang) for c in ("reason_1", "reason_2", "reason_3")
                   if c in r and isinstance(r[c], str)]
        rows.append({"user_id": str(r[id_col]), "probability_pct": _r(r["churn_probability"] * 100, 2),
                     "risk_decile": int(r["risk_decile"]), "segment": segment_name(r["segment"], ctx.lang),
                     "top_reasons_raising_risk": reasons})
    return {"source": f"session batch '{ctx.batch['name']}'", "synthetic_demo_data": bool(ctx.batch["synthetic"]),
            "customers": rows}


def risk_by(ctx: Context, column: str) -> dict:
    df = _batch_df(ctx)
    if column not in RISK_BY_COLUMNS:
        raise ToolInputError(f"column must be one of {RISK_BY_COLUMNS}.")
    key = df[column]
    if column == "segment":
        key = key.map(lambda s: segment_name(s, ctx.lang))
    key = key.astype("string").fillna("Not recorded")
    g = df.groupby(key.to_numpy())["churn_probability"].agg(["size", "mean", "sum"]).sort_values("mean",
                                                                                                  ascending=False)
    g = g[g["size"] >= 1].head(20)
    return {"source": f"session batch '{ctx.batch['name']}'", "synthetic_demo_data": bool(ctx.batch["synthetic"]),
            "column": column,
            "levels": [{"level": str(k), "customers": int(v["size"]), "mean_probability_pct": _r(v["mean"] * 100, 2),
                        "expected_churners": _r(v["sum"], 1)} for k, v in g.iterrows()]}


def run_what_if(ctx: Context, base: str = "last_customer", changes: dict | None = None) -> dict:
    changes = changes or {}
    if not isinstance(changes, dict) or not changes:
        raise ToolInputError("changes must be a non-empty object like {\"REGULARITY\": 40}.")
    bad = [k for k in changes if k not in ACTIONABLE]
    if bad:
        raise ToolInputError(f"Only these levers can change: {ACTIONABLE}. Not allowed: {bad}.")
    config = artifacts.get_config()
    if base == "last_customer":
        if not ctx.last_customer:
            raise ToolInputError("No customer has been scored on Customer Check yet; use a segment key as base.")
        baseline = dict(ctx.last_customer)
    elif base in SEGMENT_KEYS.values():
        sid = [k for k, v in SEGMENT_KEYS.items() if v == base][0]
        vals, _ = presets.preset(sid, artifacts.get_segment_profiles(), artifacts.get_optional("eda_summary"))
        baseline = {"user_id": "WHAT-IF", "REGION": "DAKAR", "TENURE": config["tenure_order"][-1], "MRG": "NO",
                    "TOP_PACK": np.nan, **vals}
    else:
        raise ToolInputError("base must be 'last_customer' or one of " + ", ".join(SEGMENT_KEYS.values()))
    scenario = dict(baseline)
    for k, v in changes.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ToolInputError(f"{k} must be a number.") from None
        if v < 0 or (k == "REGULARITY" and v > 90):
            raise ToolInputError(f"{k} out of range.")
        scenario[k] = v
    frame = pd.DataFrame([baseline, scenario])
    frame["user_id"] = ["baseline", "scenario"]
    p0, p1 = scoring.score_rows(frame, artifacts.get_bundle())
    return {"source": "deployed LightGBM model (what-if)", "base": base, "changes": changes,
            "baseline_probability_pct": _r(p0 * 100, 2), "scenario_probability_pct": _r(p1 * 100, 2),
            "change_pp": _r((p1 - p0) * 100, 2),
            "caveat": "Model sensitivity for one customer, holding everything else fixed; not a causal forecast."}


# ---------------------------------------------------------------------------
# Registry: schema + implementation
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable


def _obj(props: dict | None = None, required: list | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}


TOOLS: list[Tool] = [
    Tool("portfolio_overview", "Portfolio KPIs (customers, churn rate, estimated churners, ROC-AUC), every "
         "segment's size, churn rate, share of churners and revenue, plus computed executive insights.",
         _obj(), portfolio_overview),
    Tool("segment_profile", "Full profile of one behavioural segment with recommended actions. Segment names: "
         "Silent / Inactive, High-Value Active, Data-Engaged, Light Active, Low-Engagement.",
         _obj({"name": {"type": "string", "description": "Segment name or key"}}, ["name"]), segment_profile),
    Tool("model_card", "Facts about the deployed churn model: AUC, rows, features, trees, top drivers, "
         "limitations, whether risk thresholds are approved.", _obj(), model_card),
    Tool("batch_summary", "Aggregate summary of the customer batch scored in this session (counts, mean "
         "probability, expected churners, by segment, exploratory bands, expected revenue at risk).",
         _obj(), batch_summary),
    Tool("top_customers", f"The n riskiest customers in the session batch (n between 1 and {MAX_TOP}): ID, "
         "probability, decile, segment and the reasons raising their risk.",
         _obj({"n": {"type": "integer", "minimum": 1, "maximum": MAX_TOP}}, ["n"]), top_customers),
    Tool("risk_by", "Average churn probability and expected churners in the session batch, grouped by a column.",
         _obj({"column": {"type": "string", "enum": RISK_BY_COLUMNS}}, ["column"]), risk_by),
    Tool("run_what_if", "Re-score one customer after changing behaviour levers. base is 'last_customer' (the "
         "customer from Customer Check) or a segment key (silent, high_value, data_engaged, light_active, "
         f"low_engagement). changes maps lever names {ACTIONABLE} to new numeric values.",
         _obj({"base": {"type": "string"},
               "changes": {"type": "object", "additionalProperties": {"type": "number"}}}, ["base", "changes"]),
         run_what_if),
]


def register(tool: Tool) -> None:
    """Add a tool (used by later modules such as drift or experiments)."""
    if all(t.name != tool.name for t in TOOLS):
        TOOLS.append(tool)


def by_name(name: str) -> Tool | None:
    return next((t for t in TOOLS if t.name == name), None)


def api_schemas() -> list[dict]:
    """Tool definitions for the Messages API (inputs stream eagerly; validated before use)."""
    return [{"name": t.name, "description": t.description, "input_schema": t.schema, "eager_input_streaming": True}
            for t in TOOLS]


def call(name: str, args, ctx: Context) -> tuple[dict, bool]:
    """Run a tool safely. Returns (result, is_error)."""
    tool = by_name(name)
    if tool is None:
        return {"error": f"Unknown tool {name}"}, True
    if not isinstance(args, dict):
        return {"error": "Tool input must be a JSON object."}, True
    allowed = set(tool.schema.get("properties", {}))
    extra = set(args) - allowed
    missing = [k for k in tool.schema.get("required", []) if k not in args]
    if extra or missing:
        return {"error": f"Invalid arguments. Unexpected: {sorted(extra)}; missing: {missing}."}, True
    try:
        return tool.fn(ctx, **args), False
    except ToolInputError as exc:
        return {"error": str(exc)}, True
    except Exception as exc:  # never crash the chat on a tool failure
        return {"error": f"{type(exc).__name__}: {exc}"}, True


# Tools over exported artifacts, drift, revenue and experiments (registers itself).
from app.copilot import tools_extra  # noqa: E402,F401
