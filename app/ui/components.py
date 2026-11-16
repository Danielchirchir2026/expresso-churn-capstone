"""Reusable presentation components (HTML rendered with ``st.html``)."""

from __future__ import annotations

import html
import math

import streamlit as st

from app.i18n import t

TONE_TAG = {"info": "ui.tone.info", "good": "ui.tone.good", "warn": "ui.tone.warn", "risk": "ui.tone.risk"}


def _e(value) -> str:
    return html.escape(str(value))


def raw_html(markup: str) -> None:
    """Render HTML that contains SVG.

    ``st.html`` sanitises with DOMPurify's HTML-only profile, which strips
    SVG; markdown with unsafe HTML keeps it. Markup must be a single line.
    """
    st.markdown(markup.replace("\n", " "), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    parts = []
    if eyebrow:
        parts.append(f'<div class="hub-eyebrow">{_e(eyebrow)}</div>')
    parts.append(f'<h1 class="hub-page-title">{_e(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="hub-page-sub">{_e(subtitle)}</p>')
    st.html("".join(parts))


def section_header(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{_e(subtitle)}</p>" if subtitle else ""
    st.html(f'<div class="hub-section"><h3>{_e(title)}</h3>{sub}</div>')


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def kpi_card(label: str, value: str, delta: str | None = None, help: str | None = None,
             tone: str = "neutral", foot: str | None = None, render: bool = True) -> str:
    """One KPI card. ``tone`` colours the delta: good / bad / neutral.

    The delta always carries an arrow and text, so meaning is never colour alone.
    """
    help_html = (f'<span class="hub-help" title="{_e(help)}" aria-label="{_e(help)}">i</span>'
                 if help else "")
    delta_html = ""
    if delta:
        arrow = {"good": "▲ ", "bad": "▼ ", "neutral": ""}.get(tone, "")
        delta_html = f'<div class="hub-kpi-delta {tone}">{arrow}{_e(delta)}</div>'
    foot_html = f'<div class="hub-kpi-foot">{_e(foot)}</div>' if foot else ""
    card = (f'<div class="hub-card" role="group" aria-label="{_e(label)}: {_e(value)}">'
            f'<div class="hub-kpi-label">{_e(label)}{help_html}</div>'
            f'<div class="hub-kpi-value">{_e(value)}</div>{delta_html}{foot_html}</div>')
    if render:
        st.html(card)
    return card


def kpi_row(cards: list[dict], cols: int | None = None) -> None:
    """Render several KPI cards in a responsive grid."""
    cols = cols or min(len(cards), 4)
    inner = "".join(kpi_card(render=False, **c) for c in cards)
    st.html(f'<div class="hub-grid" style="--cols:{cols}">{inner}</div>')


# ---------------------------------------------------------------------------
# Callouts and empty states
# ---------------------------------------------------------------------------
def insight_callout(text: str, tone: str = "info", tag: str | None = None) -> None:
    tag = tag or t(TONE_TAG.get(tone, "ui.tone.info"))
    st.html(f'<div class="hub-callout {tone}"><span class="tag">{_e(tag)}</span>'
            f'<div class="body">{_e(text)}</div></div>')


def empty_state(message: str, detail: str | None = None, command: str | None = None,
                icon: str = "◌", action_label: str | None = None, action_page: str | None = None,
                key: str | None = None) -> bool:
    """Friendly placeholder. Returns True when its action button was clicked."""
    cmd = f'<div class="sub"><code>{_e(command)}</code></div>' if command else ""
    sub = f'<div class="sub">{_e(detail)}</div>' if detail else ""
    st.html(f'<div class="hub-empty"><div class="ico" aria-hidden="true">{_e(icon)}</div>'
            f'<div class="msg">{_e(message)}</div>{sub}{cmd}</div>')
    if action_label and action_page:
        st.page_link(action_page, label=action_label, icon=":material/arrow_forward:")
        return False
    if action_label:
        return st.button(action_label, key=key or f"empty_{abs(hash(message))}", type="primary")
    return False


def synthetic_banner() -> None:
    st.html(f'<div class="hub-synthetic">⚠ {_e(t("ui.synthetic_banner"))}</div>')


def badge(text: str, kind: str = "muted") -> str:
    return f'<span class="hub-badge {kind}"><span class="dot"></span>{_e(text)}</span>'


# ---------------------------------------------------------------------------
# Probability gauge (animated SVG, honours prefers-reduced-motion)
# ---------------------------------------------------------------------------
def prob_gauge(p: float, reference: float | None = None, label: str | None = None,
               color: str | None = None) -> None:
    """Semicircle gauge for a probability ``p`` in [0, 1].

    ``reference`` (0-1) draws a tick, e.g. the portfolio churn rate.
    """
    p = min(max(float(p), 0.0), 1.0)
    r, cx, cy = 80.0, 100.0, 100.0
    length = math.pi * r
    off = length * (1 - p)
    accent = color or "var(--hub-accent)"
    ref_svg, ref_html = "", ""
    if reference is not None:
        theta = math.pi * (1 - min(max(reference, 0.0), 1.0))
        x1, y1 = cx + (r - 13) * math.cos(theta), cy - (r - 13) * math.sin(theta)
        x2, y2 = cx + (r + 13) * math.cos(theta), cy - (r + 13) * math.sin(theta)
        ref_svg = (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="currentColor" '
                   f'stroke-width="3" stroke-linecap="round" opacity=".9"/>')
        ref_html = (f'<div style="text-align:center;font-size:.8rem;opacity:.75;margin-top:-.2rem">'
                    f'<span style="display:inline-block;width:3px;height:.8rem;background:currentColor;'
                    f'vertical-align:-2px;margin-right:.35rem;border-radius:2px"></span>'
                    f'{_e(t("ui.gauge.portfolio"))} {reference * 100:.2f}%</div>')
    lbl = label or t("ui.gauge.label")
    svg = (
        f'<svg class="hub-gauge" viewBox="0 0 200 128" role="img" '
        f'aria-label="{_e(lbl)}: {p * 100:.1f}%">'
        f'<path d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="currentColor" stroke-opacity=".14" '
        f'stroke-width="16" stroke-linecap="round"/>'
        f'<path class="arc" d="M20,100 A80,80 0 0 1 180,100" fill="none" stroke="{accent}" stroke-width="16" '
        f'stroke-linecap="round" style="--len:{length:.2f};--off:{off:.2f}"/>'
        f'{ref_svg}'
        f'<text x="100" y="88" text-anchor="middle" font-size="30" font-weight="800" fill="currentColor">'
        f'{p * 100:.1f}%</text>'
        f'<text x="100" y="108" text-anchor="middle" font-size="10" fill="currentColor" opacity=".7">{_e(lbl)}</text>'
        f'<text x="20" y="122" text-anchor="middle" font-size="9" fill="currentColor" opacity=".6">0%</text>'
        f'<text x="180" y="122" text-anchor="middle" font-size="9" fill="currentColor" opacity=".6">100%</text>'
        f'</svg>'
    )
    raw_html(f'<div style="color:inherit">{svg}{ref_html}</div>')


def churn_ring(share: float, label_top: str, label_bottom: str, color_a: str, color_b: str) -> str:
    """Animated donut used in the Home hero (SVG). Returns HTML."""
    r = 42.0
    c = 2 * math.pi * r
    length = c * min(max(share, 0.0), 1.0)
    return (
        f'<div class="hub-ring-wrap" role="img" aria-label="{_e(label_top)} {_e(label_bottom)}">'
        f'<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="{r}" fill="none" stroke="{color_b}" '
        f'stroke-width="11"/><circle class="seg" cx="50" cy="50" r="{r}" fill="none" stroke="{color_a}" '
        f'stroke-width="11" style="--len:{length:.2f};--c:{c:.2f}"/></svg>'
        f'<div class="hub-ring-label"><b>{_e(label_top)}</b><span>{_e(label_bottom)}</span></div></div>'
    )
