"""Offline Copilot: keyword-matched answers built from the same tools.

Used when no ANTHROPIC_API_KEY is configured (or the API is unreachable),
so the page still answers common questions in a demo. Every figure comes
from the tool functions, i.e. from the artifacts or the session batch.
"""

from __future__ import annotations

import unicodedata

from app.copilot import tools as T
from app.core import fmt
from app.i18n import t

TOPICS = [
    ("drift", ["drift", "shift", "psi", "data quality", "derive", "qualite des donnees", "stable"]),
    ("experiment", ["a/b", "ab test", "sample size", "experiment", "how many customers", "echantillon", "experience",
                    "test a/b"]),
    ("comparison", ["baseline", "logistic", "compare the model", "other model", "leaderboard", "comparaison",
                    "reference"]),
    ("prioritise", ["priorit", "focus", "first", "which segment", "cibl", "concentr", "quel segment", "d'abord"]),
    ("silent", ["silent", "inactive", "silencieux", "inactif", "dormant"]),
    ("batch", ["batch", "upload", "my file", "summar", "lot", "fichier", "resum"]),
    ("top", ["riskiest", "top customer", "who should we call", "call list", "plus risque", "appeler", "liste d'appel"]),
    ("auc", ["roc", "auc", "accura", "how good", "trust", "fiab", "performance", "precis"]),
    ("drivers", ["driver", "why do", "feature", "factor", "important", "facteur", "pourquoi", "variable"]),
    ("threshold", ["threshold", "seuil", "band", "high risk", "risque eleve", "classe"]),
    ("revenue", ["revenue", "money", "fcfa", "value", "revenu", "argent", "valeur"]),
    ("segments", ["segment", "groups", "groupe"]),
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def match(question: str) -> str:
    q = _norm(question)
    best, score = "help", 0
    for topic, words in TOPICS:
        hits = sum(1 for w in words if w in q)
        if hits > score:
            best, score = topic, hits
    return best


def answer(question: str, ctx: T.Context) -> tuple[str, list]:
    """Return (markdown answer, sources) for a question."""
    topic = match(question)
    sources: list = []

    def use(tool: str, **args):
        result, err = T.call(tool, args, ctx)
        sources.append((tool, args, err))
        return None if err else result

    lang = ctx.lang

    def pct(v, d=1):
        return fmt.pct(v, d, lang)

    def num(v, d=0):
        return fmt.num(v, d, lang)

    def money(v):
        return fmt.money(v, 0, lang)

    if topic == "prioritise":
        po = use("portfolio_overview")
        segs = sorted(po["segments"], key=lambda s: s["pct_of_revenue"] * s["churn_rate_pct"], reverse=True)
        worst = max(po["segments"], key=lambda s: s["churn_rate_pct"])
        text = t("copilot.faq.prioritise", lang, value=segs[0]["segment"], rev=pct(segs[0]["pct_of_revenue"], 1),
                 rate=pct(segs[0]["churn_rate_pct"], 2), worst=worst["segment"],
                 worst_rate=pct(worst["churn_rate_pct"], 1), share=pct(worst["pct_of_churners"], 0))
    elif topic == "silent":
        sp = use("segment_profile", name="silent")
        mc = use("model_card")
        text = t("copilot.faq.silent", lang, rate=pct(sp["churn_rate_pct"], 1),
                 portfolio=pct(sp["portfolio_churn_rate_pct"], 2), share=pct(sp["pct_of_churners"], 0),
                 days=num(sp["median_active_days_of_90"]), d1=mc["top_drivers_by_gain"][0]["driver"],
                 d2=mc["top_drivers_by_gain"][1]["driver"]) + "\n\n" + "\n".join(
            f"- {a}" for a in sp["recommended_actions"][:3])
    elif topic in ("batch", "top"):
        if not ctx.batch:
            text = t("copilot.faq.no_batch", lang)
        elif topic == "batch":
            bs = use("batch_summary")
            seg = max(bs["by_segment"], key=lambda s: s["mean_probability_pct"])
            text = t("copilot.faq.batch", lang, n=num(bs["customers"]), mean=pct(bs["mean_probability_pct"], 2),
                     exp=num(bs["expected_churners"]), seg=seg["segment"],
                     segp=pct(seg["mean_probability_pct"], 1))
            if bs.get("expected_monthly_revenue_at_risk_fcfa") is not None:
                text += " " + t("copilot.faq.batch_rev", lang, rev=money(bs["expected_monthly_revenue_at_risk_fcfa"]))
            if bs["synthetic_demo_data"]:
                text += "\n\n_" + t("ui.synthetic_banner", lang) + "_"
        else:
            tc = use("top_customers", n=5)
            lines = [f"- **{c['user_id']}**: {pct(c['probability_pct'], 1)} · {c['segment']} · "
                     + (", ".join(c["top_reasons_raising_risk"]) or "–") for c in tc["customers"]]
            text = t("copilot.faq.top", lang) + "\n\n" + "\n".join(lines)
            if tc["synthetic_demo_data"]:
                text += "\n\n_" + t("ui.synthetic_banner", lang) + "_"
    elif topic == "auc":
        mc = use("model_card")
        auc = mc["validation_roc_auc"]
        text = t("copilot.faq.auc", lang, auc=num(auc, 4), pct=pct(auc * 100, 1),
                 rows=num(mc["training_rows"]))
    elif topic == "drivers":
        mc = use("model_card")
        lines = [f"{i}. {d['driver']} ({pct(d['gain_share_pct'], 1)})" for i, d in
                 enumerate(mc["top_drivers_by_gain"][:5], 1)]
        text = t("copilot.faq.drivers", lang) + "\n\n" + "\n".join(lines)
    elif topic == "threshold":
        use("model_card")
        text = t("copilot.faq.threshold", lang)
        ev = use("evaluation_at_threshold", threshold=0.5)
        if ev:
            text += " " + t("copilot.faq.threshold_eval", lang, p=pct(ev["precision_pct"], 0), r=pct(ev["recall_pct"], 0))
    elif topic == "revenue":
        po = use("portfolio_overview")
        rar = use("revenue_at_risk")
        text = t("copilot.faq.revenue", lang, share=pct(po["high_value_active_revenue_share_pct"], 1))
        if rar:
            text += " " + t("copilot.faq.rar", lang, v=money(rar["portfolio_monthly_revenue_at_risk_fcfa"]))
            if rar.get("batch"):
                text += " " + t("copilot.faq.batch_rev", lang, rev=money(rar["batch"]["expected_monthly_revenue_at_risk_fcfa"]))
    elif topic == "drift":
        if not ctx.batch:
            text = t("copilot.faq.no_batch", lang)
        else:
            dr = use("drift_report")
            if dr is None:
                text = t("copilot.faq.drift_missing", lang)
            else:
                top = dr["columns"][0]
                text = t(f"copilot.faq.drift_{dr['overall']}", lang, col=top["column"], psi=num(top["psi"], 3),
                         red=dr["counts"].get("red", 0), amber=dr["counts"].get("amber", 0))
    elif topic == "experiment":
        ss = use("sample_size", reduction=20)
        text = t("copilot.faq.experiment", lang, n=num(ss["customers_per_group"]), base=pct(ss["baseline_churn_pct"], 2),
                 target=pct(ss["target_churn_pct"], 2))
    elif topic == "comparison":
        mc = use("model_comparison")
        if mc is None:
            text = t("copilot.faq.comparison_missing", lang)
        else:
            lines = [f"- **{m['model']}**: ROC-AUC {num(m['roc_auc'], 4)} · PR-AUC {num(m['pr_auc'], 4)}"
                     for m in mc["models"]]
            text = t("copilot.faq.comparison", lang) + "\n\n" + "\n".join(lines)
    elif topic == "segments":
        po = use("portfolio_overview")
        lines = [f"- **{s['segment']}**: {pct(s['pct_customers'], 1)} · churn {pct(s['churn_rate_pct'], 2)} · "
                 f"{t('copilot.faq.rev_word', lang)} {pct(s['pct_of_revenue'], 1)}" for s in po["segments"]]
        text = t("copilot.faq.segments", lang) + "\n\n" + "\n".join(lines)
    else:
        text = t("copilot.faq.help", lang)
    return text, sources
