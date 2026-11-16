"""Static checks on translations and shared UI assets."""

import re
from pathlib import Path

from app.i18n import catalog

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [p for p in list(ROOT.glob("app/**/*.py")) + list(ROOT.glob("app_pages/*.py")) + [ROOT / "Home.py"]]
KEY_RE = re.compile(r"""\bt[l]?\(\s*["']([a-z0-9_]+(?:\.[a-z0-9_]+)+)["']""")


def test_english_and_french_have_the_same_keys():
    en, fr = set(catalog("en")), set(catalog("fr"))
    assert en - fr == set(), f"missing in fr: {sorted(en - fr)[:20]}"
    assert fr - en == set(), f"missing in en: {sorted(fr - en)[:20]}"


def test_every_literal_key_used_in_code_exists():
    en = catalog("en")
    missing = []
    for path in SOURCES:
        for key in KEY_RE.findall(path.read_text(encoding="utf-8")):
            if key not in en and not any(k.startswith(key + ".") for k in en):
                missing.append(f"{path.relative_to(ROOT)}: {key}")
    assert not missing, missing


def test_placeholders_match_between_languages():
    en, fr = catalog("en"), catalog("fr")
    field = re.compile(r"\{(\w+)\}")
    bad = [k for k, v in en.items() if isinstance(v, str) and k in fr and isinstance(fr[k], str)
           and set(field.findall(v)) != set(field.findall(fr[k]))]
    assert not bad, bad


def test_stylesheet_survives_st_html_sanitiser():
    # DOMPurify drops <style> blocks whose text contains "<" followed by a
    # letter or slash, which silently removes every style in the file.
    css = (ROOT / "app" / "ui" / "style.css").read_text(encoding="utf-8")
    assert "<" not in css
