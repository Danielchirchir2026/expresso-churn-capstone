"""Preprocessing and prediction tests for the deployed LightGBM bundle."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.prediction import predict_customer_churn, prepare_customer_features
from tests import reference_prediction_dd86793 as reference

ROOT = Path(__file__).resolve().parents[1]


def test_feature_schema_matches_artifact(edge_rows, config):
    features = prepare_customer_features(edge_rows, config)
    with open(ROOT / "artifacts" / "feature_columns.json", encoding="utf-8") as fh:
        artifact_columns = json.load(fh)

    assert list(features.columns) == config["feature_columns"]
    assert list(features.columns) == artifact_columns
    assert features.shape == (len(edge_rows), 69)


def test_no_infinite_values(edge_rows, config):
    features = prepare_customer_features(edge_rows, config)
    numeric = features.to_numpy(dtype="float64")
    assert not np.isinf(numeric).any()


def test_probabilities_are_valid(edge_rows, model, config):
    result = predict_customer_churn(edge_rows, model, config)
    assert list(result.columns) == ["user_id", "churn_probability"]
    assert result["churn_probability"].between(0, 1).all()
    assert result["churn_probability"].notna().all()


def test_edge_case_semantics(edge_rows, config):
    f = prepare_customer_features(edge_rows, config).reset_index(drop=True)
    other_freq = config["top_pack_frequency"]["Other"]

    # Row 2: missing everything that may be missing
    assert f.loc[1, "is_missing_MONTANT"] == 1
    assert f.loc[1, "MONTANT_MISSING"] == 1
    assert f.loc[1, "REGION_MISSING"] == 1
    assert f.loc[1, "MISSING_COUNT"] == len(config["missing_source_columns"])

    # Row 3: unseen region falls into REGION_Missing, rare pack into "Other"
    assert f.loc[2, "REGION_Missing"] == 1
    assert f.loc[2, [c for c in f.columns if c.startswith("REGION_") and c not in
                     ("REGION_Missing", "REGION_MISSING")]].sum() == 0
    assert np.isclose(f.loc[2, "TOP_PACK_FREQ"], other_freq)

    # Row 4: zero recharges and zero revenue give NaN ratios, never inf
    assert np.isnan(f.loc[3, "REVENUE_PER_RECHARGE"])
    assert np.isnan(f.loc[3, "DATA_SHARE"])

    # Row 5: padded region is stripped, unknown tenure is coded -1,
    # unknown pack falls back to the "Other" frequency
    assert f.loc[4, "REGION_KOLDA"] == 1
    assert f.loc[4, "TENURE_ORDINAL"] == -1
    assert np.isclose(f.loc[4, "TOP_PACK_FREQ"], other_freq)


def test_parity_with_original_dd86793_logic(edge_rows, model, config):
    """The fixed module must reproduce the original deployment exactly."""
    new = prepare_customer_features(edge_rows, config)
    old = reference.prepare_customer_features(edge_rows, config)
    pd.testing.assert_frame_equal(new, old)

    new_p = model.predict_proba(new)[:, 1]
    old_p = model.predict_proba(old)[:, 1]
    np.testing.assert_array_equal(new_p, old_p)


def test_duplicate_ids_rejected(edge_rows, config):
    dup = pd.concat([edge_rows, edge_rows.head(1)], ignore_index=True)
    try:
        prepare_customer_features(dup, config)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate IDs should raise")
