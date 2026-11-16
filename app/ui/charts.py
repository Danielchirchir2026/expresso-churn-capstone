"""Plotly conventions shared by every chart.

* One fixed colour per segment (``segment_color``), validated for colour-
  blind separation in both themes.
* Recessive hairline grids, thin rounded bars, hover on every mark.
* ``show()`` renders the chart, its "So what?" caption, an ``alt`` text
  for screen readers and a "View as table" alternative.
"""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.segments import SEGMENT_COLORS
from app.i18n import t

INK = {
    "dark": dict(primary="#E6EDF7", secondary="#AEBBD0", muted="#7F8BA3", grid="#1F2A40", axis="#33415E",
                 surface="#0B1220"),
    "light": dict(primary="#0B1220", secondary="#45526B", muted="#6B778C", grid="#E3E8F0", axis="#C3CCDA",
                  surface="#F7F9FC"),
}
ACCENT = {"dark": "#2DD4BF", "light": "#0F766E"}
# Diverging pair: risk-up (red) vs risk-down (blue); neutral grey midpoint.
RISK_UP = {"dark": "#e66767", "light": "#e34948"}
RISK_DOWN = {"dark": "#3987e5", "light": "#2a78d6"}
NEUTRAL = {"dark": "#55607A", "light": "#B4BDCB"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
# Sequential single-hue blue ramp; dark mode runs dark->light so large
# values gain contrast against the navy surface.
SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]
SEQ_DARK = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]
SERIES = {
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
}


def mode() -> str:
    try:
        value = st.context.theme.type
    except Exception:
        value = None
    return value if value in ("light", "dark") else "dark"


def segment_color(seg_id) -> str:
    return SEGMENT_COLORS[mode()][int(seg_id)]


def segment_color_map(names: dict) -> dict:
    """{display_name: colour} for px-style colour maps."""
    return {name: segment_color(sid) for sid, name in names.items()}


def accent() -> str:
    return ACCENT[mode()]


def series(i: int) -> str:
    return SERIES[mode()][i % 8]


def risk_up() -> str:
    return RISK_UP[mode()]


def risk_down() -> str:
    return RISK_DOWN[mode()]


def neutral() -> str:
    return NEUTRAL[mode()]


def sequential() -> list[str]:
    return SEQ_DARK if mode() == "dark" else SEQ_LIGHT


def ink(role: str = "primary") -> str:
    return INK[mode()][role]


def presenting() -> bool:
    return bool(st.session_state.get("present_mode", False))


def style(fig: go.Figure, height: int = 380, legend: bool = True, ygrid: bool = True,
          xgrid: bool = False, title: str | None = None, pct_y: bool = False,
          pct_x: bool = False) -> go.Figure:
    """Apply the house style to a figure."""
    m = mode()
    h = int(height * (1.3 if presenting() else 1.0))
    fig.update_layout(
        height=h,
        margin=dict(l=8, r=12, t=48 if title else 28, b=8),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        font=dict(family="Inter, system-ui, sans-serif", color=INK[m]["secondary"],
                  size=15 if presenting() else 13),
        hoverlabel=dict(font=dict(family="Inter, system-ui, sans-serif", size=13)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None,
                    font=dict(size=12)),
        bargap=0.35,
        barcornerradius=4,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=xgrid, gridcolor=INK[m]["grid"], gridwidth=1, zeroline=False,
                     linecolor=INK[m]["axis"], tickfont=dict(color=INK[m]["muted"]),
                     title_font=dict(color=INK[m]["secondary"]))
    fig.update_yaxes(showgrid=ygrid, gridcolor=INK[m]["grid"], gridwidth=1, zeroline=False,
                     linecolor=INK[m]["axis"], tickfont=dict(color=INK[m]["muted"]),
                     title_font=dict(color=INK[m]["secondary"]))
    if pct_y:
        fig.update_yaxes(ticksuffix="%")
    if pct_x:
        fig.update_xaxes(ticksuffix="%")
    return fig


def show(fig: go.Figure, so_what: str, data: pd.DataFrame | None = None, key: str | None = None,
         alt: str | None = None) -> None:
    """Render a chart + its one-line business takeaway + accessible table."""
    st.plotly_chart(
        fig,
        width="stretch",
        theme="streamlit",
        key=key,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2}},
        alt=alt or so_what,
    )
    st.html(f'<div class="hub-sowhat"><b>{html.escape(t("ui.so_what"))}</b> {html.escape(so_what)}</div>')
    if data is not None and not presenting():
        with st.expander(t("ui.view_table"), icon=":material/table_view:"):
            st.dataframe(data, hide_index=True, width="stretch")
