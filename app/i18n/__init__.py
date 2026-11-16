"""Tiny i18n layer: every UI string lives in en.json / fr.json.

Keys are dotted paths into the nested JSON (e.g. ``t("home.title")``).
Missing French keys fall back to English, then to the key itself, so a
missing translation never crashes a page.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parent
LANGUAGES = {"en": "English", "fr": "Français"}
DEFAULT_LANG = "en"


def _flatten(node, prefix=""):
    flat = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


@lru_cache(maxsize=None)
def catalog(lang: str) -> dict:
    path = I18N_DIR / f"{lang}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return _flatten(json.load(fh))


def current_lang() -> str:
    try:
        import streamlit as st

        lang = st.session_state.get("lang", DEFAULT_LANG)
    except Exception:  # outside a Streamlit run
        lang = DEFAULT_LANG
    return lang if lang in LANGUAGES else DEFAULT_LANG


def t(key: str, lang: str | None = None, **kwargs):
    """Translate ``key``; format with ``kwargs`` when given."""
    lang = lang or current_lang()
    value = catalog(lang).get(key)
    if value is None:
        value = catalog(DEFAULT_LANG).get(key, key)
    if kwargs and isinstance(value, str):
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return value
    return value


def tl(key: str, lang: str | None = None) -> list:
    """Translate a key whose value is a list of strings."""
    value = t(key, lang)
    return value if isinstance(value, list) else [value]
