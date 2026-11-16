"""Chunked scoring, contributions, deciles, segments and action rules."""

import numpy as np
import pandas as pd

from app.core import explain, scoring
from app.core.segments import assign_segments
from app.utils.prediction import prepare_customer_features
from tests import reference_prediction_dd86793 as reference


def test_chunked_batch_matches_original_logic(edge_rows, bundle, model, config):
    """Refactored, chunked path == original dd86793 preprocessing + predict_proba."""
    out = scoring.score_batch(edge_rows, bundle, chunk_size=2)
    expected = model.predict_proba(reference.prepare_customer_features(edge_rows, config))[:, 1]
    np.testing.assert_allclose(out["churn_probability"].to_numpy(), expected, rtol=0, atol=1e-12)
    assert list(out["user_id"]) == list(edge_rows["user_id"])


def test_contributions_reconstruct_probability(edge_rows, model, config):
    feats = prepare_customer_features(edge_rows, config)
    contrib = scoring.contributions(feats, model)
    assert contrib.shape == (len(edge_rows), len(config["feature_columns"]) + 1)
    prob = 1 / (1 + np.exp(-contrib.sum(axis=1)))
    np.testing.assert_allclose(prob, scoring.predict_proba(feats, model), atol=1e-9)

    grouped, bias = scoring.grouped_contributions(contrib, config["feature_columns"], config["raw_source_columns"])
    np.testing.assert_allclose(grouped.sum(axis=1).to_numpy() + bias, contrib.sum(axis=1), atol=1e-9)
    assert "REGULARITY" in grouped.columns and "LOG1P_MONTANT" not in grouped.columns


def test_reason_codes_are_risk_raising_only(edge_rows, bundle):
    out = scoring.score_batch(edge_rows, bundle, chunk_size=3)
    for k in (1, 2, 3):
        vals = out[f"reason_{k}_logodds"].dropna()
        assert (vals > 0).all()
    assert out["segment"].between(0, 4).all()


def test_risk_deciles():
    p = np.linspace(0.99, 0.0, 20)
    d = scoring.risk_deciles(p)
    assert d[0] == 1 and d[-1] == 10
    assert (np.diff(d) >= 0).all()
    assert np.bincount(d)[1:].tolist() == [2] * 10


def test_bands():
    keys = scoring.bands([0.05, 0.2, 0.6], low=0.1, high=0.5)
    assert keys.tolist() == ["low", "medium", "high"]


def test_silent_rule_assigns_segment_zero(config, bundle):
    row = {c: np.nan for c in config["raw_source_columns"]}
    row.update(user_id="s", REGION="DAKAR", TENURE="K > 24 month", MRG="NO", REGULARITY=2)
    df = pd.DataFrame([row])[[config["id_column"]] + config["raw_source_columns"]]
    seg = assign_segments(df, prepare_customer_features(df, config), bundle)
    assert seg.tolist() == [0]


def test_next_best_action_rules():
    low = explain.ActionContext(prob=0.02, portfolio_rate=0.1875, top_up={"REGULARITY"}, missing_count=0,
                                region_missing=False, revenue=1000, high_value_revenue=8600)
    assert explain.next_best_actions(low) == ["low_risk"]
    vip = explain.ActionContext(prob=0.4, portfolio_rate=0.1875, top_up={"REGULARITY", "DATA_VOLUME"},
                                missing_count=1, region_missing=False, revenue=12000, high_value_revenue=8600)
    acts = explain.next_best_actions(vip)
    assert acts[0] == "vip_call" and "activity" in acts and len(acts) <= 3
