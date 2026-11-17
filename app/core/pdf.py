"""Executive Brief PDF (fpdf2), with the hero chart drawn as native vectors.

The bubble chart is drawn with fpdf2 primitives instead of a kaleido image:
kaleido 1.x needs a Chrome install, which Streamlit Community Cloud does not
provide, and vector drawing keeps the PDF small and sharp.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
from fpdf import FPDF

from app.core import fmt, portfolio
from app.core.segments import SEGMENT_COLORS, with_names
from app.i18n import t

NAVY = (27, 31, 36)  # header band: graphite (#1B1F24), name kept for the modules that import it
TEAL = (31, 91, 181)  # accent: cobalt (#1F5BB5)
INK = (27, 31, 36)
MUTED = (90, 98, 112)
GRID = (225, 228, 232)
REPLACE = {" ": " ", " ": " ", "—": "-", "–": "-", "’": "'", "‘": "'",
           "“": '"', "”": '"', "−": "-", "≥": ">=", "≤": "<=", "…": "...",
           "▲": "+", "▼": "-", "≈": "~", "œ": "oe", "Œ": "OE", "→": "->",
           "•": "-", "↑": "^"}


def clean(text) -> str:
    s = str(text)
    for a, b in REPLACE.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Brief(FPDF):
    def __init__(self, lang: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.lang = lang
        self.set_auto_page_break(True, margin=18)
        self.set_margins(15, 15, 15)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, clean(t("pdf.footer", self.lang)), align="L")
        self.set_x(-30)
        self.cell(15, 5, f"{self.page_no()}/{{nb}}", align="R")


def _kpis(pdf: Brief, cards: list[tuple[str, str]]) -> None:
    w, h, gap = 58, 17, 3
    x0, y0 = 15, pdf.get_y()
    for i, (label, value) in enumerate(cards):
        x = x0 + (i % 3) * (w + gap)
        y = y0 + (i // 3) * (h + gap)
        pdf.set_fill_color(243, 244, 246)
        pdf.set_draw_color(*GRID)
        pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=2.5)
        pdf.set_fill_color(*TEAL)
        pdf.rect(x, y + 3, 1.2, h - 6, style="F")
        pdf.set_xy(x + 4, y + 3)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(w - 6, 4, clean(label))
        pdf.set_xy(x + 4, y + 7.5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*INK)
        pdf.cell(w - 6, 8, clean(value))
    pdf.set_y(y0 + 2 * (h + gap) + 2)


def _bubble(pdf: Brief, profiles: pd.DataFrame, pf: portfolio.Portfolio) -> None:
    """x = % customers, y = churn rate, area = % revenue, one colour per segment."""
    x0, y0, w, h = 27, pdf.get_y() + 5, 165, 60
    xmin, xmax, ymin, ymax = 12.0, 27.0, -9.0, 60.0
    sx = lambda v: x0 + (v - xmin) / (xmax - xmin) * w  # noqa: E731
    sy = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h  # noqa: E731
    pdf.set_draw_color(*GRID)
    pdf.set_line_width(0.2)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*MUTED)
    for gx in range(12, 28, 2):
        pdf.line(sx(gx), y0, sx(gx), y0 + h)
        pdf.text(sx(gx) - 2.5, y0 + h + 4, f"{gx}%")
    for gy in range(0, 61, 10):
        pdf.line(x0, sy(gy), x0 + w, sy(gy))
        pdf.text(x0 - 9, sy(gy) + 1, f"{gy}%")
    pdf.set_font("Helvetica", "", 7)
    pdf.text(x0 + 1, y0 - 1.2, clean(t("exec.axis.churn_rate", pdf.lang) + " ↑"))
    pdf.set_draw_color(*MUTED)
    pdf.set_line_width(0.3)
    pdf.line(x0, sy(pf.churn_rate), x0 + w, sy(pf.churn_rate))
    pdf.text(x0 + 1, sy(pf.churn_rate) - 1.2, clean(t("exec.portfolio_line", pdf.lang, rate=fmt.pct(pf.churn_rate, 2,
                                                                                                    pdf.lang))))
    max_rev = float(profiles["pct_of_revenue"].max())
    for _, r in profiles.sort_values("pct_of_revenue", ascending=False).iterrows():
        radius = max(math.sqrt(max(r["pct_of_revenue"], 0) / max_rev) * 8.5, 1.6)
        cx, cy = sx(r["pct_customers"]), sy(min(r["churn_rate"], ymax))
        pdf.set_fill_color(*_rgb(SEGMENT_COLORS["light"][int(r["segment"])]))
        pdf.set_draw_color(255, 255, 255)
        pdf.set_line_width(0.5)
        pdf.ellipse(cx - radius, cy - radius, 2 * radius, 2 * radius, style="DF")
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*INK)
        label = clean(r["segment_name"])
        lw = pdf.get_string_width(label)
        if r["churn_rate"] > 10 or r["pct_of_revenue"] < 5:      # small or high: label above
            lx, ly = cx - lw / 2, cy - radius - 1.5
        elif radius > 6:                                          # big bubble near the axis: label left
            lx, ly = cx - radius - lw - 1.5, cy + 1
        else:                                                     # otherwise below
            lx, ly = cx - lw / 2, cy + radius + 3.5
        lx = min(max(lx, x0 + 1), x0 + w - lw - 1)
        pdf.text(lx, ly, label)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*MUTED)
    pdf.text(x0 + w / 2 - 12, y0 + h + 8.5, clean(t("exec.axis.pct_customers", pdf.lang)))
    pdf.set_y(y0 + h + 11)


def executive_brief(profiles_raw: pd.DataFrame, meta: dict, lang: str = "en", batch: dict | None = None) -> bytes:
    profiles = with_names(profiles_raw, lang)
    pf = portfolio.compute(profiles, meta)
    pdf = Brief(lang)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Header band
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 30, style="F")
    pdf.set_xy(15, 8)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, clean(t("pdf.title", lang)))
    pdf.set_xy(15, 17)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(200, 208, 220)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pdf.cell(0, 6, clean(t("pdf.subtitle", lang, date=today)))
    pdf.set_y(37)

    _kpis(pdf, [
        (t("kpi.total_customers", lang), fmt.num(pf.total_customers, 0, lang)),
        (t("kpi.churn_rate", lang), fmt.pct(pf.churn_rate, 2, lang)),
        (t("kpi.est_churners", lang), fmt.num(pf.est_churners, 0, lang)),
        (t("kpi.roc_auc", lang), fmt.num(pf.roc_auc, 4, lang)),
        (t("kpi.revenue_concentration", lang), fmt.pct(pf.revenue_concentration, 1, lang)),
        (t("kpi.churners_silent", lang), fmt.pct(pf.churners_from_silent, 1, lang)),
    ])

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, clean(t("exec.bubble.title", lang)), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 4, clean(t("exec.bubble.sub", lang)), new_x="LMARGIN", new_y="NEXT")
    _bubble(pdf, profiles, pf)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, clean(t("exec.insights.title", lang)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    for _, text in portfolio.executive_insights(profiles, pf, lang):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*INK)
        pdf.set_x(15)
        pdf.multi_cell(0, 4.3, clean("- " + text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.6)

    # Segment table
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, clean(t("exec.table.title", lang)), new_x="LMARGIN", new_y="NEXT")
    cols = [(t("exec.col.segment", lang), 52), (t("exec.hover.customers", lang), 30),
            (t("exec.axis.churn_rate", lang), 30), (t("exec.share.churners", lang), 34),
            (t("exec.share.revenue", lang), 34)]
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(233, 235, 239)
    for name, wd in cols:
        pdf.cell(wd, 6, clean(name), border=0, fill=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 8.5)
    for _, r in profiles.iterrows():
        vals = [r["segment_name"], fmt.num(r["customers"], 0, lang), fmt.pct(r["churn_rate"], 2, lang),
                fmt.pct(r["pct_of_churners"], 1, lang), fmt.pct(r["pct_of_revenue"], 1, lang)]
        for (_, wd), v in zip(cols, vals):
            pdf.cell(wd, 5.5, clean(v))
        pdf.ln(5.5)

    if batch:
        df = batch["df"]
        p = df["churn_probability"]
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 6, clean(t("pdf.batch_title", lang, name=batch["name"])), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        line = t("pdf.batch_line", lang, n=fmt.num(len(df), 0, lang), mean=fmt.pct(p.mean() * 100, 2, lang),
                 exp=fmt.num(p.sum(), 0, lang))
        if batch.get("synthetic"):
            line += " " + t("ui.synthetic_banner", lang)
        pdf.multi_cell(0, 4.6, clean(line), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 3.8, clean(t("pdf.provenance", lang, created=meta.get("created_utc", "")[:10])),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
