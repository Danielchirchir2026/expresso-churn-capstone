"""Executive Brief PDF and data dictionary."""

import json
from pathlib import Path

import pandas as pd

from app.core.dictionary import data_dictionary
from app.core.pdf import clean, executive_brief

ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    profiles = pd.read_csv(ROOT / "artifacts" / "segment_profiles.csv")
    meta = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
    return profiles, meta


def test_pdf_builds_in_both_languages():
    profiles, meta = _inputs()
    for lang in ("en", "fr"):
        pdf = executive_brief(profiles, meta, lang)
        assert pdf[:5] == b"%PDF-" and len(pdf) > 2000


def test_pdf_with_batch_section(edge_rows, bundle):
    from app.core import scoring
    profiles, meta = _inputs()
    df = scoring.score_batch(edge_rows, bundle)
    pdf = executive_brief(profiles, meta, "en", {"df": df, "name": "edge.csv", "synthetic": True})
    assert pdf[:5] == b"%PDF-"


def test_clean_keeps_french_accents_and_drops_unsupported():
    assert clean("Équipe — 18,75\u202f%") == "Équipe - 18,75 %"
    assert clean("œuvre") == "oeuvre"


def test_data_dictionary_covers_every_raw_column(config):
    d = data_dictionary(config, "en")
    for col in [config["id_column"]] + config["raw_source_columns"]:
        assert col in set(d["column"])
    used = d.set_index("column")["used_by_model"]
    assert used["MRG"] == "no" and used["REGULARITY"] == "yes"
    assert data_dictionary(config, "fr").set_index("column").loc["REGION", "name"] == "Région"


def test_feature_groups_cover_all_69_features_once(config):
    from app.core.features import OWN_GROUPS, source_of

    groups = [source_of(f, config["raw_source_columns"]) for f in config["feature_columns"]]
    assert set(groups) <= set(config["raw_source_columns"]) | set(OWN_GROUPS)
    assert len(groups) == 69


def test_pdf_library_loaded_lazily():
    """fpdf2 costs about a second to import; pages and the Copilot load it only when a PDF is requested."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = ("import sys; import app.copilot.summary, app.copilot.tools, app.core.portfolio; "
            "assert 'fpdf' not in sys.modules, 'fpdf imported eagerly'")
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
    for page in ("executive.py", "reports.py"):
        text = (root / "app_pages" / page).read_text(encoding="utf-8")
        assert "\nfrom app.core.pdf import" not in text, page
