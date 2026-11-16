"""export_artifacts.py end to end on a small synthetic Train.csv, then the
pages that use those files, run against them (the default smoke tests cover
the same pages without the files)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.core import artifacts, demo, scoring, validation

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["eda_summary.json", "eda_breakdowns.parquet", "validation_predictions.parquet", "model_comparison.csv",
            "reference_stats.json", "export_manifest.json"]


@pytest.fixture(scope="session")
def exported(tmp_path_factory, bundle, config):
    tmp = tmp_path_factory.mktemp("export")
    df = demo.generate(20_000, pd.read_csv(ROOT / "artifacts" / "segment_profiles.csv"), config, seed=3)
    p = scoring.score_batch(validation.clean(df, config), bundle, with_reasons=False, with_segments=False)
    # Test-only labels drawn from the model's own probabilities: exercises the mechanics, not the model.
    df["CHURN"] = (np.random.default_rng(0).random(len(df)) < p["churn_probability"].to_numpy()).astype(int)
    train = tmp / "Train.csv"
    df.to_csv(train, index=False)
    out = tmp / "artifacts"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_artifacts.py"), "--train", str(train),
                           "--out", str(out), "--baseline-sample", "5000"], capture_output=True, text=True,
                          cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


def test_export_script_writes_every_file(exported):
    for name in EXPECTED:
        assert (exported / name).exists(), name
    manifest = json.loads((exported / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_mb"] < 20
    assert manifest["split"]["seed"] == 42
    preds = pd.read_parquet(exported / "validation_predictions.parquet")
    assert list(preds.columns) == ["y_true", "y_prob", "REGION", "TENURE"]  # no customer IDs
    comp = pd.read_csv(exported / "model_comparison.csv")
    assert {"Dummy (prior)", "Logistic Regression", "LightGBM (deployed)"} <= set(comp["model"])
    summary = json.loads((exported / "eda_summary.json").read_text(encoding="utf-8"))
    assert set(summary["segment_medians"]) == {"0", "1", "2", "3", "4"}


PAGES_WITH_ARTIFACTS = ["data_explorer", "problem", "methodology", "evaluation", "customer_check", "what_if"]


@pytest.mark.parametrize("page", PAGES_WITH_ARTIFACTS)
def test_pages_with_exported_artifacts(page, exported, monkeypatch):
    monkeypatch.setattr(artifacts, "OPTIONAL_DIR", exported)
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=300).run()
    at.switch_page(f"app_pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Something went wrong on this page." not in [e.value for e in at.error]
