"""Rule-based alerts raised after a batch is scored (rules stated on screen)."""

from __future__ import annotations

import pandas as pd

REGION_FACTOR = 1.25  # a region is flagged when its mean risk is 25% above the batch mean
MIN_REGION = 30       # ... and it has at least this many customers


def batch_alerts(df: pd.DataFrame, high_threshold: float, id_col: str = "user_id",
                 region_factor: float = REGION_FACTOR, min_region: int = MIN_REGION) -> dict:
    p = df["churn_probability"]
    rev = pd.to_numeric(df.get("REVENUE"), errors="coerce").fillna(0) if "REVENUE" in df else pd.Series(0, index=df.index)
    paying = rev > 0
    rev_cut = float(rev[paying].quantile(0.9)) if paying.any() else float("inf")
    vip_mask = paying & (rev >= rev_cut) & (df["risk_decile"] == 1)
    vip = df.loc[vip_mask].assign(_rev=rev[vip_mask]).sort_values("_rev", ascending=False)

    region = df["REGION"].astype("string").fillna("Not recorded") if "REGION" in df else pd.Series("All", index=df.index)
    overall = float(p.mean())
    g = pd.DataFrame({"region": region, "p": p}).groupby("region")["p"].agg(["mean", "size"])
    flagged = g[(g["size"] >= min_region) & (g["mean"] > overall * region_factor)].sort_values("mean", ascending=False)
    return {
        "above": int((p >= high_threshold).sum()),
        "above_share": float((p >= high_threshold).mean()) if len(p) else 0.0,
        "vip_count": int(vip_mask.sum()),
        "vip_revenue_cut": rev_cut,
        "vip": vip.drop(columns=["_rev"]),
        "overall_mean": overall,
        "regions": [(str(r), float(m), int(n)) for r, (m, n) in flagged.iterrows()],
    }
