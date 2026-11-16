"""Shared pytest fixtures for the Expresso churn app."""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def bundle():
    return joblib.load(ROOT / "artifacts" / "expresso_churn_bundle.joblib")


@pytest.fixture(scope="session")
def config(bundle):
    return bundle["preprocessing_config"]


@pytest.fixture(scope="session")
def model(bundle):
    return bundle["model"]


@pytest.fixture(scope="session")
def edge_rows(config):
    """Five raw rows built from raw_source_columns covering awkward cases."""
    rare_pack = config["rare_packs"][0]
    common_pack = next(
        k for k in config["top_pack_frequency"] if k not in ("MISSING", "Other")
    )
    rows = [
        # 1. Fully populated, typical active customer
        dict(user_id="u1", REGION="DAKAR", TENURE="K > 24 month", MONTANT=5000,
             FREQUENCE_RECH=10, REVENUE=5000, ARPU_SEGMENT=1667, FREQUENCE=12,
             DATA_VOLUME=3000, ON_NET=200, ORANGE=50, TIGO=10, ZONE1=1, ZONE2=2,
             MRG="NO", REGULARITY=60, TOP_PACK=common_pack, FREQ_TOP_PACK=8),
        # 2. Missing values everywhere they are allowed
        dict(user_id="u2", REGION=np.nan, TENURE="K > 24 month", MONTANT=np.nan,
             FREQUENCE_RECH=np.nan, REVENUE=np.nan, ARPU_SEGMENT=np.nan,
             FREQUENCE=np.nan, DATA_VOLUME=np.nan, ON_NET=np.nan, ORANGE=np.nan,
             TIGO=np.nan, ZONE1=np.nan, ZONE2=np.nan, MRG="NO", REGULARITY=2,
             TOP_PACK=np.nan, FREQ_TOP_PACK=np.nan),
        # 3. Unseen region and a rare TOP_PACK
        dict(user_id="u3", REGION="ATLANTIS", TENURE="E 6-9 month", MONTANT=1000,
             FREQUENCE_RECH=2, REVENUE=900, ARPU_SEGMENT=300, FREQUENCE=3,
             DATA_VOLUME=0, ON_NET=5, ORANGE=3, TIGO=0, ZONE1=np.nan, ZONE2=np.nan,
             MRG="NO", REGULARITY=15, TOP_PACK=rare_pack, FREQ_TOP_PACK=1),
        # 4. Zero recharges and zero revenue (division by zero paths)
        dict(user_id="u4", REGION="THIES", TENURE="H 15-18 month", MONTANT=0,
             FREQUENCE_RECH=0, REVENUE=0, ARPU_SEGMENT=0, FREQUENCE=0,
             DATA_VOLUME=500, ON_NET=0, ORANGE=0, TIGO=0, ZONE1=0, ZONE2=0,
             MRG="NO", REGULARITY=30, TOP_PACK="Data:490F=1GB,7d", FREQ_TOP_PACK=0),
        # 5. Unknown pack, padded region text, unknown tenure label
        dict(user_id="u5", REGION=" KOLDA ", TENURE="Z unknown", MONTANT=250,
             FREQUENCE_RECH=1, REVENUE=250, ARPU_SEGMENT=83, FREQUENCE=1,
             DATA_VOLUME=np.nan, ON_NET=1, ORANGE=np.nan, TIGO=np.nan, ZONE1=np.nan,
             ZONE2=np.nan, MRG="NO", REGULARITY=5, TOP_PACK="BRAND_NEW_PACK_2026",
             FREQ_TOP_PACK=1),
    ]
    df = pd.DataFrame(rows)
    return df[[config["id_column"]] + config["raw_source_columns"]]


@pytest.fixture(scope="session")
def exported(tmp_path_factory, bundle, config):
    """Run scripts/export_artifacts.py on a small synthetic Train.csv (test-only labels)."""
    from app.core import demo, scoring, validation

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
