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


def test_copilot_with_api_key_handles_api_errors(monkeypatch):
    """With a key the page goes online; an API failure shows a warning and falls back to the offline answer.

    The streaming call is replaced, so no request leaves the machine.
    """
    import anthropic

    try:  # the pinned SDK (2.9) uses httpx2 as its transport; older releases used httpx
        import httpx2 as httpx
    except ImportError:  # pragma: no cover
        import httpx

    from app.copilot import llm

    def failing_stream(*args, **kwargs):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        yield  # pragma: no cover - makes this a generator like the real one

    monkeypatch.setattr(llm, "stream_answer", failing_stream)
    monkeypatch.setattr("app.ui.layout.get_secret",
                        lambda name, default=None: {"ANTHROPIC_API_KEY": "sk-test-not-a-real-key"}.get(name, default))
    at = load("copilot", "en")
    assert_clean(at, "en")
    assert t("copilot.mode.offline_help", "en") not in [c.value for c in at.caption]  # online mode
    at.chat_input[0].set_value("Which segment should we prioritise?").run()
    assert_clean(at, "en")
    assert t("copilot.err.connection", "en") in [w.value for w in at.warning]
