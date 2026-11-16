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


def test_pages_do_not_call_st_stop():
    # st.stop() inside a page halts the router too, so the footer and tour
    # never render. Pages use if/else instead.
    offenders = [p.name for p in (ROOT / "app_pages").glob("*.py") if "st.stop()" in p.read_text(encoding="utf-8")]
    assert not offenders, offenders


def test_role_views_reorder_navigation():
    from app.navigation import PAGES, ROLES, ordered

    for role in ROLES:
        layout = ordered(role)
        keys = [k for _, ks in layout for k in ks]
        assert sorted(keys) == sorted(PAGES), role          # every page exactly once
        assert layout[0] == ("", ["home"])                    # Home always first
    assert ordered("executive")[1] == ("overview", ["executive", "problem", "data_explorer"])
    assert ordered("retention")[1][0] == "act" and ordered("retention")[1][1][0] == "retention"
    assert ordered("analyst")[1][0] == "trust" and ordered("analyst")[1][1][0] == "model"


def test_guided_tour_walks_every_step():
    from streamlit.testing.v1 import AppTest

    from app.ui.layout import tour_steps

    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=300).run()
    at.button(key="home_tour").click().run()
    steps = tour_steps()
    for i in range(len(steps)):
        assert at.session_state["tour_step"] == i
        assert not at.exception, [e.value for e in at.exception]
        at.button(key="tour_next").click().run()
    assert at.session_state["tour_step"] is None


def test_presentation_mode_and_french_render():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=300)
    at.session_state["present_mode"] = True
    at.session_state["lang"] = "fr"
    at.run()
    at.switch_page("app_pages/executive.py").run()
    assert not at.exception
    assert any(b.key == "present_exit" for b in at.button)
    assert any("Tableau de bord direction" in h.value for h in at.get("html"))
