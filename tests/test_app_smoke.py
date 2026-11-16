"""AppTest smoke tests: every registered page loads through the router."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.i18n import t
from app.navigation import PAGES, page_path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = str(ROOT / "Home.py")
BUILT = [k for k in PAGES if page_path(k)]


def load(key: str, lang: str = "en") -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=240)
    at.session_state["lang"] = lang
    at.run()
    if key != "home":
        at.switch_page(page_path(key)).run()
    return at


def assert_clean(at: AppTest, lang: str) -> None:
    assert not at.exception, [e.value for e in at.exception]
    boundary = t("ui.error.title", lang)
    errors = [e.value for e in at.error]
    assert boundary not in errors, errors


@pytest.mark.parametrize("key", BUILT)
def test_page_loads_english(key):
    assert_clean(load(key, "en"), "en")


@pytest.mark.parametrize("key", BUILT)
def test_page_loads_french(key):
    assert_clean(load(key, "fr"), "fr")
