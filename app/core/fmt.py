"""Locale-aware number formatting (English and French conventions)."""

from __future__ import annotations

import math

from app.i18n import current_lang

NBSP = " "  # narrow no-break space used as French thousands separator


def _is_missing(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except TypeError:
        return False


def num(x, decimals: int = 0, lang: str | None = None) -> str:
    if _is_missing(x):
        return "–"
    lang = lang or current_lang()
    s = f"{x:,.{decimals}f}"
    if lang == "fr":
        s = s.replace(",", "§").replace(".", ",").replace("§", NBSP)
    return s


def pct(x, decimals: int = 1, lang: str | None = None, already_pct: bool = True) -> str:
    """Format a percentage. ``x`` is 0-100 when ``already_pct`` else 0-1."""
    if _is_missing(x):
        return "–"
    value = x if already_pct else x * 100
    lang = lang or current_lang()
    body = num(value, decimals, lang)
    return f"{body}{NBSP}%" if lang == "fr" else f"{body}%"


def pp(x, decimals: int = 1, lang: str | None = None) -> str:
    """Signed percentage-point difference."""
    if _is_missing(x):
        return "–"
    sign = "+" if x > 0 else ("−" if x < 0 else "±")
    return f"{sign}{num(abs(x), decimals, lang)} pp"


def money(x, decimals: int = 0, lang: str | None = None, unit: str = "FCFA") -> str:
    if _is_missing(x):
        return "–"
    return f"{num(x, decimals, lang)} {unit}"


def compact(x, lang: str | None = None) -> str:
    """Short form for large numbers: 2.15M / 2,15 M."""
    if _is_missing(x):
        return "–"
    lang = lang or current_lang()
    for div, suf in ((1e9, "B" if lang == "en" else " Md"), (1e6, "M" if lang == "en" else " M"),
                     (1e3, "k" if lang == "en" else " k")):
        if abs(x) >= div:
            return f"{num(x / div, 2, lang)}{suf}"
    return num(x, 0, lang)
