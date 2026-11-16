"""Turn LightGBM contributions into plain sentences and suggested actions.

Reason codes come from the model (``pred_contrib``). Next best actions are
business rules layered on top of those reasons; they are suggestions, not
model output, and the UI labels them that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from app.core import fmt
from app.core.features import friendly
from app.i18n import t

STRONG, MODERATE = 1.0, 0.3  # |log-odds| thresholds for wording


def strength(value: float) -> str:
    a = abs(value)
    return "strong" if a >= STRONG else ("moderate" if a >= MODERATE else "slight")


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def describe_value(group: str, raw: dict, feats: pd.Series, config: dict) -> str:
    """Short phrase describing this customer's value for a source group."""
    v = raw.get(group)
    if group == "REGULARITY":
        return t("reason.val.REGULARITY", v=fmt.num(feats.get("REGULARITY", float("nan"))))
    if group == "REGION":
        if _missing(v):
            return t("reason.val.REGION_missing")
        if str(v).strip() not in config["region_values"]:
            return t("reason.val.REGION_unseen", v=v)
        return t("reason.val.REGION", v=str(v).strip())
    if group == "TENURE":
        return t("reason.val.TENURE", v=v)
    if group == "TOP_PACK":
        return t("reason.val.TOP_PACK_missing") if _missing(v) else t("reason.val.TOP_PACK", v=v)
    if group == "MISSING_COUNT":
        return t("reason.val.MISSING_COUNT", v=fmt.num(feats.get("MISSING_COUNT", 0)),
                 n=len(config["missing_source_columns"]))
    if group in ("REVENUE_PER_RECHARGE", "DATA_SHARE"):
        fv = feats.get(group)
        if fv is None or (isinstance(fv, float) and math.isnan(fv)):
            return t("reason.val.ratio_missing", name=friendly(group))
        return t("reason.val.numeric", name=friendly(group), v=fmt.num(fv, 1 if fv < 100 else 0))
    if _missing(v):
        return t("reason.val.missing", name=friendly(group))
    try:
        return t("reason.val.numeric", name=friendly(group), v=fmt.num(float(v)))
    except (TypeError, ValueError):
        return t("reason.val.text", name=friendly(group), v=v)


def reason_sentences(grouped: pd.Series, raw: dict, feats: pd.Series, config: dict,
                     n: int = 5) -> list[tuple[str, str]]:
    """[(direction 'up'|'down', sentence)] for the n largest contributions."""
    out = []
    top = grouped[grouped.abs() > 1e-6].head(n)
    for i, (group, value) in enumerate(top.items()):
        direction = "up" if value > 0 else "down"
        text = t(f"reason.sentence.{direction}", what=describe_value(group, raw, feats, config),
                 strength=t(f"reason.strength.{strength(value)}"))
        if i == 0:
            text += " " + t("reason.biggest")
        out.append((direction, text))
    return out


def reason_label(group: str | None) -> str:
    """Compact label used in batch tables (risk-raising reasons only)."""
    if group is None or (isinstance(group, float) and math.isnan(group)):
        return ""
    return f"↑ {friendly(group)}"


# ---------------------------------------------------------------------------
# Next best action (rule-based business suggestions)
# ---------------------------------------------------------------------------
@dataclass
class ActionContext:
    prob: float
    portfolio_rate: float  # 0-1
    top_up: set
    missing_count: int
    region_missing: bool
    revenue: float
    high_value_revenue: float  # median revenue of the High-Value Active segment


RECHARGE_GROUPS = {"FREQUENCE_RECH", "MONTANT", "REVENUE_PER_RECHARGE", "FREQUENCE", "REVENUE", "ARPU_SEGMENT"}


def next_best_actions(ctx: ActionContext, limit: int = 3) -> list[str]:
    """Keys of suggested actions, most relevant first (see i18n 'nba.*')."""
    if ctx.prob < ctx.portfolio_rate / 2:
        return ["low_risk"]
    keys = []
    if (not math.isnan(ctx.revenue) and ctx.revenue >= ctx.high_value_revenue
            and ctx.prob >= ctx.portfolio_rate):
        keys.append("vip_call")
    if ctx.missing_count >= 8:
        keys.append("dormant")
    if "REGION" in ctx.top_up and ctx.region_missing:
        keys.append("verify_record")
    if "REGULARITY" in ctx.top_up:
        keys.append("activity")
    if ctx.top_up & RECHARGE_GROUPS:
        keys.append("recharge")
    if ctx.top_up & {"DATA_VOLUME", "DATA_SHARE"}:
        keys.append("data")
    if "ON_NET" in ctx.top_up:
        keys.append("onnet")
    if ctx.top_up & {"ORANGE", "TIGO"}:
        keys.append("allnet")
    if ctx.top_up & {"TOP_PACK", "FREQ_TOP_PACK"}:
        keys.append("pack")
    if not keys:
        keys.append("monitor")
    seen, ordered = set(), []
    for k in keys:
        if k not in seen:
            ordered.append(k)
            seen.add(k)
    return ordered[:limit]


def build_context(prob: float, grouped: pd.Series, raw: dict, feats: pd.Series, portfolio_rate: float,
                  profiles: pd.DataFrame) -> ActionContext:
    up = grouped[grouped > 0.1].head(4)
    rev = raw.get("REVENUE")
    try:
        rev = float(rev)
    except (TypeError, ValueError):
        rev = float("nan")
    hv = float(profiles.set_index("segment").loc[1, "median_revenue"])
    return ActionContext(
        prob=prob,
        portfolio_rate=portfolio_rate,
        top_up=set(up.index),
        missing_count=int(feats.get("MISSING_COUNT", 0) or 0),
        region_missing=_missing(raw.get("REGION")),
        revenue=rev,
        high_value_revenue=hv,
    )
