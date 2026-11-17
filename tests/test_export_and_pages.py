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


PAGES_WITH_ARTIFACTS = ["data_explorer", "problem", "methodology", "evaluation", "customer_check", "what_if", "drift"]


@pytest.mark.parametrize("page", PAGES_WITH_ARTIFACTS)
def test_pages_with_exported_artifacts(page, exported, monkeypatch):
    monkeypatch.setattr(artifacts, "OPTIONAL_DIR", exported)
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=300).run()
    at.switch_page(f"app_pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Something went wrong on this page." not in [e.value for e in at.error]


def test_only_deployed_model_marked_optimistic(exported, tmp_path, monkeypatch):
    """With a holdout refit, only the deployed model's own score is flagged optimistic, not the holdout row
    (whose name also contains "deployed settings")."""
    import shutil

    folder = tmp_path / "artifacts"
    shutil.copytree(exported, folder)
    comp = pd.read_csv(folder / "model_comparison.csv")
    dep = comp[comp["model"] == "LightGBM (deployed)"].iloc[0]
    holdout = dep.copy()
    holdout["model"] = "LightGBM (holdout refit, deployed settings)"
    holdout["roc_auc"] = dep["roc_auc"] - 0.002
    pd.concat([comp, holdout.to_frame().T], ignore_index=True).to_csv(folder / "model_comparison.csv", index=False)

    monkeypatch.setattr(artifacts, "OPTIONAL_DIR", folder)
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=300).run()
    at.switch_page("app_pages/evaluation.py").run()
    assert not at.exception, [e.value for e in at.exception]
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    trace = spec["data"][0]
    labelled = [y for y, text in zip(trace["y"], trace["text"]) if "optimistic" in text]
    assert labelled == ["LightGBM (deployed)"], labelled
