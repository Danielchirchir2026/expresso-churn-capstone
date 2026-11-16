"""PSI and drift report, checked by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from app.core import drift


def test_psi_hand_computed():
    # (0.6 - 0.5) ln(0.6/0.5) + (0.4 - 0.5) ln(0.4/0.5) = 0.018232 + 0.022314 = 0.040546
    assert drift.psi([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.040546, abs=1e-6)
    assert drift.psi([0.25] * 4, [0.25] * 4) == 0.0
    # Shifting all mass to one bin is a large drift
    assert drift.psi([0.5, 0.5], [1.0, 0.0]) > drift.AMBER


def test_status_thresholds():
    assert drift.status(0.05) == "green"
    assert drift.status(0.15) == "amber"
    assert drift.status(0.30) == "red"
    assert drift.status(float("nan")) == "na"


def test_numeric_and_categorical_bins_include_missing_and_unseen():
    ref_num = {"edges": [10.0], "proportions": [0.4, 0.4], "missing_rate": 0.2}
    e, a = drift.numeric_bins(pd.Series([1, 5, 20, 30, np.nan]), ref_num)
    assert e.tolist() == pytest.approx([0.4, 0.4, 0.2])
    assert a.tolist() == pytest.approx([0.4, 0.4, 0.2])  # identical -> PSI 0
    assert drift.psi(e, a) == pytest.approx(0.0)

    ref_cat = {"proportions": {"DAKAR": 0.5, "THIES": 0.3}, "missing_rate": 0.2}
    e, a, unseen = drift.categorical_bins(pd.Series(["DAKAR", "ATLANTIS", None, "THIES"]), ref_cat)
    assert e.tolist() == pytest.approx([0.5, 0.3, 0.0, 0.2])
    assert a.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert unseen == ["ATLANTIS"]


def test_report_flags_shifted_column():
    reference = {"rows": 100, "numeric": {"X": {"edges": [10.0], "proportions": [0.5, 0.5], "missing_rate": 0.0}},
                 "categorical": {}}
    same = drift.report(pd.DataFrame({"X": [1, 20] * 50}), reference, {})
    moved = drift.report(pd.DataFrame({"X": [50] * 100}), reference, {})
    assert same["overall"] == "green" and moved["overall"] == "red"
    assert math.isfinite(moved["rows"][0]["psi"])


def test_batch_alerts_rules():
    from app.core.alerts import batch_alerts

    n = 200
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "churn_probability": np.r_[np.full(40, 0.9), np.full(160, 0.05)],
        "risk_decile": np.r_[np.ones(20, int), np.full(180, 5)],
        "REVENUE": np.r_[np.full(20, 10_000.0), np.full(180, 100.0)],
        "REGION": ["KOLDA"] * 40 + ["DAKAR"] * 160,
    })
    al = batch_alerts(df, high_threshold=0.5)
    assert al["above"] == 40
    assert al["vip_count"] == 20                      # top revenue decile AND risk decile 1
    assert [r[0] for r in al["regions"]] == ["KOLDA"]  # 0.9 > 1.25 x batch mean (0.22)


def test_drift_report_on_exported_reference(exported, edge_rows, config):
    import json

    reference = json.loads((exported / "reference_stats.json").read_text(encoding="utf-8"))
    rep = drift.report(edge_rows, reference, config)
    assert {r["column"] for r in rep["rows"]} >= {"REGULARITY", "REGION", "TENURE"}
    unseen = {r["column"]: r["unseen"] for r in rep["rows"]}
    assert "ATLANTIS" in unseen["REGION"]
