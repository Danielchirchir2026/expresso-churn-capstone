"""Synthetic demo customers, clearly labelled as such everywhere they appear.

Values are sampled around the segment medians in ``segment_profiles.csv``
(share of customers, revenue, recharges, active days, data use, off-net
share). Pack popularity comes from the model config. When the exported
EDA breakdowns are available, tenure and region follow the real training
mix. Nothing generated here is ever shown as a real-world figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lognormal(rng, median: float, sigma: float, n: int) -> np.ndarray:
    median = max(float(median), 1.0)
    return rng.lognormal(np.log(median), sigma, n)


def generate(n: int, profiles: pd.DataFrame, config: dict, seed: int = 42, breakdowns: pd.DataFrame | None = None,
             messy: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = profiles.set_index("segment")
    seg_ids = p.index.to_numpy()
    seg = rng.choice(seg_ids, size=n, p=(p["pct_customers"] / p["pct_customers"].sum()).to_numpy())

    cols = {c: np.full(n, np.nan) for c in ["MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE",
                                            "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2",
                                            "REGULARITY", "FREQ_TOP_PACK"]}
    packs = [k for k in config["top_pack_frequency"] if k not in ("MISSING", "Other")]
    weights = np.array([config["top_pack_frequency"][k] for k in packs], dtype=float)
    top_pack = np.full(n, None, dtype=object)
    data_scale = float(p["median_data_volume"].max()) or 1000.0

    for s in seg_ids:
        idx = np.flatnonzero(seg == s)
        k = len(idx)
        if k == 0:
            continue
        r = p.loc[s]
        days = np.clip(rng.normal(r["median_active_days"], 6 if s == 0 else 12, k), 1, 90).round()
        cols["REGULARITY"][idx] = days
        if s == 0:  # silent: usage fields stay empty
            continue
        rev = _lognormal(rng, r["median_revenue"] if r["median_revenue"] > 0 else 300, 0.7, k).round()
        rech = np.maximum(1, rng.poisson(max(r["median_recharges"], 1), k))
        cols["REVENUE"][idx] = rev
        cols["MONTANT"][idx] = (rev * rng.uniform(0.85, 1.15, k)).round(-1)
        cols["ARPU_SEGMENT"][idx] = (rev / 3).round()
        cols["FREQUENCE_RECH"][idx] = rech
        cols["FREQUENCE"][idx] = rech + rng.poisson(1.5, k)
        uses_data = rng.random(k) < r["data_users"] / 100
        dv_median = r["median_data_volume"] if r["median_data_volume"] > 0 else data_scale / 4
        cols["DATA_VOLUME"][idx] = np.where(uses_data, _lognormal(rng, dv_median, 1.0, k).round(), np.nan)
        calls = _lognormal(rng, max(rev.mean() / 25, 5), 0.9, k)
        off = np.clip(r["offnet_share"] / 100 + rng.normal(0, 0.08, k), 0, 0.95)
        has_calls = rng.random(k) < 0.85
        cols["ON_NET"][idx] = np.where(has_calls, (calls * (1 - off)).round(), np.nan)
        cols["ORANGE"][idx] = np.where(has_calls, (calls * off * 0.8).round(), np.nan)
        cols["TIGO"][idx] = np.where(has_calls & (rng.random(k) < 0.6), (calls * off * 0.2).round(), np.nan)
        cols["ZONE1"][idx] = np.where(rng.random(k) < 0.08, rng.poisson(3, k), np.nan)
        cols["ZONE2"][idx] = np.where(rng.random(k) < 0.06, rng.poisson(3, k), np.nan)
        has_pack = rng.random(k) < 0.8
        top_pack[idx] = np.where(has_pack, rng.choice(packs, size=k, p=weights / weights.sum()), None)
        cols["FREQ_TOP_PACK"][idx] = np.where(has_pack, np.maximum(1, rng.poisson(max(r["median_recharges"], 1) * 0.6, k)),
                                              np.nan)

    regions = [r for r in config["region_values"] if r != "MISSING"]
    tenures = list(config["tenure_order"])
    region_p = tenure_p = None
    if breakdowns is not None and not breakdowns.empty:
        reg = breakdowns[breakdowns["dimension"] == "REGION"]
        reg = reg[reg["level"].isin(regions)]
        if len(reg):
            regions, region_p = list(reg["level"]), (reg["count"] / reg["count"].sum()).to_numpy()
        ten = breakdowns[breakdowns["dimension"] == "TENURE"]
        ten = ten[ten["level"].isin(tenures)]
        if len(ten):
            tenures, tenure_p = list(ten["level"]), (ten["count"] / ten["count"].sum()).to_numpy()
    if tenure_p is None:  # no exported mix: weight towards long tenure, labelled synthetic
        tenure_p = np.linspace(1, 8, len(tenures)) ** 3
        tenure_p = tenure_p / tenure_p.sum()
    region = rng.choice(regions, size=n, p=region_p)
    region = np.where(rng.random(n) < np.where(seg == 0, 0.7, 0.15), None, region)

    df = pd.DataFrame({
        "user_id": [f"DEMO-{i:06d}" for i in range(1, n + 1)],
        "REGION": region,
        "TENURE": rng.choice(tenures, size=n, p=tenure_p),
        **{c: cols[c] for c in ["MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE", "DATA_VOLUME",
                                "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2"]},
        "MRG": "NO",
        "REGULARITY": cols["REGULARITY"],
        "TOP_PACK": top_pack,
        "FREQ_TOP_PACK": cols["FREQ_TOP_PACK"],
    })
    df = df[["user_id"] + list(config["raw_source_columns"])]

    if messy and n >= 20:
        # A handful of realistic problems to demonstrate the validation report.
        df.loc[3, "user_id"] = df.loc[2, "user_id"]
        df.loc[5, "REGION"] = "ATLANTIS"
        df["MONTANT"] = df["MONTANT"].astype(object)
        df.loc[7, "MONTANT"] = "12k"
        df.loc[9, "REGULARITY"] = 120
        df.loc[11, "TENURE"] = "L > 36 month"
    return df
