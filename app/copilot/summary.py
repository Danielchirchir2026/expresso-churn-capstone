"""One-page executive summary of the current state, written from tool facts only.

``facts()`` gathers everything through the same aggregate-only tools the
Copilot uses. Without an API key ``template_summary()`` writes the summary
from fixed templates; with a key ``stream_llm_summary()`` asks Claude to write
it from the facts alone. Both can be exported as a PDF.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Iterator

from app.copilot import tools as T
from app.core import fmt
from app.i18n import t


def facts(ctx: T.Context) -> tuple[dict, list]:
    """(facts, sources) for the summary; missing sources are simply skipped."""
    out, sources = {}, []
    calls = [("portfolio", "portfolio_overview", {}), ("model", "model_card", {}),
             ("comparison", "model_comparison", {}), ("revenue", "revenue_at_risk", {})]
    if ctx.batch:
        calls += [("batch", "batch_summary", {}), ("top", "top_customers", {"n": 5}), ("drift", "drift_report", {})]
    for key, name, args in calls:
        result, err = T.call(name, args, ctx)
        if not err:
            out[key] = result
            sources.append((name, args, False))
    return out, sources


def recommended_actions(f: dict, lang: str) -> list[str]:
    """Rule-based recommendations; each cites the numbers that triggered it."""
    acts = []
    segs = f.get("portfolio", {}).get("segments", [])
    if segs:
        worst = max(segs, key=lambda s: s["churn_rate_pct"])
        value = max(segs, key=lambda s: s["pct_of_revenue"] * s["churn_rate_pct"])
        acts.append(t("summary.act.reactivate", lang, seg=worst["segment"], rate=fmt.pct(worst["churn_rate_pct"], 1, lang)))
        acts.append(t("summary.act.protect", lang, seg=value["segment"], rev=fmt.pct(value["pct_of_revenue"], 1, lang)))
    b = f.get("batch")
    if b:
        acts.append(t("summary.act.calls", lang, n=fmt.num(b["expected_churners"], 0, lang)))
    d = f.get("drift")
    if d and d["overall"] != "green":
        acts.append(t("summary.act.drift", lang, n=d["counts"].get("red", 0) + d["counts"].get("amber", 0)))
    if not f.get("model", {}).get("risk_thresholds_approved", True):
        acts.append(t("summary.act.thresholds", lang))
    acts.append(t("summary.act.test", lang))
    return acts[:6]


def template_summary(f: dict, lang: str) -> str:
    p, m = f["portfolio"], f["model"]
    lines = [f"### {t('summary.h.portfolio', lang)}",
             t("summary.portfolio", lang, n=fmt.num(p["total_customers"], 0, lang),
               rate=fmt.pct(p["overall_churn_rate_pct"], 2, lang), churners=fmt.num(p["estimated_churners"], 0, lang),
               rev=fmt.pct(p["high_value_active_revenue_share_pct"], 1, lang),
               silent=fmt.pct(p["silent_inactive_share_of_churners_pct"], 1, lang)),
             f"### {t('summary.h.model', lang)}",
             t("summary.model", lang, auc=fmt.num(m["validation_roc_auc"], 4, lang),
               d1=m["top_drivers_by_gain"][0]["driver"], d2=m["top_drivers_by_gain"][1]["driver"])]
    if f.get("comparison"):
        lr = next((x for x in f["comparison"]["models"] if "Logistic" in x["model"]), None)
        if lr:
            lines.append(t("summary.baseline", lang, lr=fmt.num(lr["roc_auc"], 4, lang)))
    rev = f.get("revenue")
    if rev:
        lines.append(t("summary.revenue", lang, v=fmt.money(rev["portfolio_monthly_revenue_at_risk_fcfa"], 0, lang)))
    b = f.get("batch")
    lines.append(f"### {t('summary.h.batch', lang)}")
    if b:
        lines.append(t("summary.batch", lang, n=fmt.num(b["customers"], 0, lang),
                       mean=fmt.pct(b["mean_probability_pct"], 2, lang), exp=fmt.num(b["expected_churners"], 0, lang)))
        if b.get("synthetic_demo_data"):
            lines.append(f"_{t('ui.synthetic_banner', lang)}_")
        d = f.get("drift")
        lines.append(t(f"summary.drift_{d['overall']}", lang, red=d["counts"].get("red", 0),
                       amber=d["counts"].get("amber", 0)) if d else t("summary.drift_na", lang))
    else:
        lines.append(t("summary.no_batch", lang))
    lines.append(f"### {t('summary.h.actions', lang)}")
    lines += [f"- {a}" for a in recommended_actions(f, lang)]
    lines.append(f"_{t('ui.decision_support', lang)}_")
    return "\n\n".join(lines)


def stream_llm_summary(client, model: str, effort: str, lang: str, f: dict) -> Iterator[str]:
    """Claude writes the summary from the facts only (single call, streamed)."""
    from app.copilot.llm import FALLBACK_BETA, FALLBACK_MODELS, LANG_NAMES

    system = (
        "You write one-page executive summaries for Expresso Senegal's management about customer churn. "
        "Use ONLY the JSON facts provided; never add, estimate or round away numbers; if something is missing, "
        "say it is not available. Structure: four short sections with markdown '###' headings: Portfolio, Model, "
        "Latest batch and data health, Recommended actions (3-6 bullets). Keep it under 350 words, plain business "
        "language, decision-support tone (probabilities are not verdicts; risk thresholds are not approved). "
        "If a fact says synthetic_demo_data is true, say so. "
        f"Write in {LANG_NAMES.get(lang, 'English')}."
    )
    facts_json = json.dumps({**f, "rule_based_recommendations": recommended_actions(f, lang)}, ensure_ascii=False,
                            default=str)
    kwargs = dict(model=model, max_tokens=4000, system=system, output_config={"effort": effort},
                  messages=[{"role": "user", "content": f"Facts:\n{facts_json}"}])
    cm = (client.beta.messages.stream(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
          if model in FALLBACK_MODELS else client.messages.stream(**kwargs))
    with cm as stream:
        for event in stream:
            if event.type == "text":
                yield event.text
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n_" + t("copilot.refusal", lang) + "_"


def summary_pdf(markdown_text: str, lang: str) -> bytes:
    from fpdf import FPDF  # imported here: fpdf2 is slow to import and only needed for the download

    from app.core.pdf import INK, MUTED, NAVY, clean

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 26, style="F")
    pdf.set_xy(16, 8)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 7, clean(t("summary.pdf_title", lang)))
    pdf.set_xy(16, 16)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(173, 233, 223)
    pdf.cell(0, 5, clean(t("summary.pdf_sub", lang, date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))))
    pdf.set_y(32)
    for block in markdown_text.split("\n"):
        line = block.strip()
        if not line:
            pdf.ln(1.5)
            continue
        if line.startswith("#"):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 6, clean(line.lstrip("#").strip()), new_x="LMARGIN", new_y="NEXT")
            continue
        italic = line.startswith("_") and line.endswith("_")
        text = line.strip("_")
        bullet = text.startswith(("- ", "* "))
        if bullet:
            text = "- " + text[2:]
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        pdf.set_font("Helvetica", "I" if italic else "", 9.5)
        pdf.set_text_color(*(MUTED if italic else INK))
        pdf.multi_cell(0, 4.8, clean(text), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
