"""Batch validation: friendly, row-level messages instead of tracebacks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUMERIC = ["MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE", "DATA_VOLUME", "ON_NET", "ORANGE",
           "TIGO", "ZONE1", "ZONE2", "REGULARITY", "FREQ_TOP_PACK"]
SAMPLE = 8  # rows listed per issue


@dataclass
class Issue:
    severity: str  # "block" | "warn" | "info"
    code: str  # i18n key suffix under batch.issue.*
    params: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)  # 1-based file row numbers (header = row 1)
    count: int = 0


@dataclass
class Report:
    n_rows: int
    n_cols: int
    issues: list[Issue]

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "block"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.blocking


def _rows(mask: pd.Series) -> tuple[list, int]:
    idx = np.flatnonzero(mask.to_numpy())
    return [int(i) + 2 for i in idx[:SAMPLE]], int(len(idx))


def validate(df: pd.DataFrame, config: dict) -> Report:
    id_col = config["id_column"]
    required = [id_col] + list(config["raw_source_columns"])
    issues: list[Issue] = []

    if df.empty:
        issues.append(Issue("block", "empty"))
        return Report(0, df.shape[1], issues)

    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(Issue("block", "missing_columns", {"cols": ", ".join(missing)}, count=len(missing)))
    extra = [c for c in df.columns if c not in required and c != config.get("target_column", "CHURN")]
    if extra:
        issues.append(Issue("info", "extra_columns", {"cols": ", ".join(map(str, extra[:12]))}, count=len(extra)))
    if missing:
        return Report(len(df), df.shape[1], issues)

    ids = df[id_col]
    m = ids.isna()
    if not pd.api.types.is_numeric_dtype(ids):
        m = m | (ids.astype("string").str.strip() == "")
    if m.any():
        rows, n = _rows(m)
        issues.append(Issue("block", "missing_ids", rows=rows, count=n))
    d = ids.duplicated(keep="first") & ~m
    if d.any():
        rows, n = _rows(d)
        issues.append(Issue("block", "duplicate_ids", {"example": str(ids[d].iloc[0])}, rows=rows, count=n))

    for col in NUMERIC:
        raw = df[col]
        if pd.api.types.is_numeric_dtype(raw):
            coerced, bad = raw, pd.Series(False, index=raw.index)
        else:
            coerced = pd.to_numeric(raw, errors="coerce")
            bad = coerced.isna() & raw.notna() & (raw.astype("string").str.strip() != "")
        if bad.any():
            rows, n = _rows(bad)
            issues.append(Issue("warn", "not_numeric", {"col": col, "example": str(raw[bad].iloc[0])[:30]},
                                rows=rows, count=n))
        neg = coerced < 0
        if neg.any():
            rows, n = _rows(neg)
            issues.append(Issue("warn", "negative", {"col": col}, rows=rows, count=n))
    reg = pd.to_numeric(df["REGULARITY"], errors="coerce")
    out = (reg < 0) | (reg > 90)
    if out.any():
        rows, n = _rows(out)
        issues.append(Issue("warn", "regularity_range", rows=rows, count=n))
    if reg.isna().any():
        rows, n = _rows(reg.isna())
        issues.append(Issue("warn", "regularity_missing", rows=rows, count=n))

    region = df["REGION"].astype("string").str.strip()
    unseen = region.notna() & ~region.isin(config["region_values"])
    if unseen.any():
        rows, n = _rows(unseen)
        vals = ", ".join(sorted(region[unseen].dropna().unique().astype(str))[:5])
        issues.append(Issue("warn", "unknown_region", {"values": vals}, rows=rows, count=n))
    tenure = df["TENURE"].astype("string").str.strip()
    bad_t = ~tenure.isin(config["tenure_order"])
    if bad_t.any():
        rows, n = _rows(bad_t)
        vals = ", ".join(sorted(tenure[bad_t].fillna("<empty>").unique().astype(str))[:5])
        issues.append(Issue("warn", "unknown_tenure", {"values": vals}, rows=rows, count=n))
    pack = df["TOP_PACK"].astype("string").str.strip()
    known = set(config["top_pack_frequency"]) | set(config["rare_packs"])
    new_pack = pack.notna() & ~pack.isin(known)
    if new_pack.any():
        rows, n = _rows(new_pack)
        issues.append(Issue("info", "unknown_pack", {"values": ", ".join(sorted(pack[new_pack].unique().astype(str))[:3])},
                            rows=rows, count=n))
    mrg = df["MRG"].astype("string").str.strip()
    if (mrg.notna() & (mrg != "NO")).any():
        issues.append(Issue("info", "mrg_values", count=int((mrg.notna() & (mrg != "NO")).sum())))

    usage = df[NUMERIC[:-2] + ["FREQ_TOP_PACK"]].apply(pd.to_numeric, errors="coerce")
    silent = usage.isna().all(axis=1)
    if silent.any():
        issues.append(Issue("info", "silent_rows", {"share": f"{silent.mean() * 100:.1f}"}, count=int(silent.sum())))
    return Report(len(df), df.shape[1], issues)


def clean(df: pd.DataFrame, config: dict, drop_missing_ids: bool = True, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply the fixes the user accepted; keep only the columns the model needs."""
    id_col = config["id_column"]
    out = df.copy()
    if drop_missing_ids:
        ids = out[id_col]
        out = out[~(ids.isna() | (ids.astype("string").str.strip() == ""))]
    if drop_duplicates:
        out = out[~out[id_col].duplicated(keep="first")]
    keep = [id_col] + list(config["raw_source_columns"])
    if config.get("target_column", "CHURN") in out.columns:
        keep.append(config["target_column"])
    out = out[keep].reset_index(drop=True)
    # Values that are not numbers are treated as not recorded (as the
    # validation report says), so the missing-value indicators stay consistent.
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def template(config: dict) -> pd.DataFrame:
    return pd.DataFrame(columns=[config["id_column"]] + list(config["raw_source_columns"]))
