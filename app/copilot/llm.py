"""LLM-backed Copilot: grounded system prompt + streaming tool loop.

Uses the official Anthropic Python SDK. The API key comes only from
``st.secrets["ANTHROPIC_API_KEY"]``; the model from ``COPILOT_MODEL``
(default ``claude-opus-5-5``). Nothing here hard-codes a key.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterator

from app.copilot import tools as T
from app.core import artifacts, portfolio
from app.core.features import friendly, global_importance, grouped_importance
from app.core.segments import with_names
from app.i18n import catalog

if TYPE_CHECKING:  # the SDK is slow to import; load it only when a key is configured
    import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOOL_TURNS = 6
# Models that accept server-side refusal fallbacks ("default" routing).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
LANG_NAMES = {"en": "English", "fr": "French"}


def system_prompt(lang: str) -> str:
    """Stable grounding context: model card facts, segment table, drivers, glossary."""
    meta = artifacts.get_metadata()
    profiles = with_names(artifacts.get_segment_profiles(), "en")
    pf = portfolio.compute(profiles, meta)
    config = artifacts.get_config()
    imp = grouped_importance(global_importance(artifacts.get_model(), tuple(config["feature_columns"]),
                                               tuple(config["raw_source_columns"])))
    seg_lines = "\n".join(
        f"- {r.segment_name}: {int(r.customers):,} customers ({r.pct_customers:.1f}%), churn {r.churn_rate:.2f}%, "
        f"{r.pct_of_churners:.1f}% of churners, {r.pct_of_revenue:.1f}% of revenue, median active days "
        f"{r.median_active_days:.0f}, data users {r.data_users:.1f}%"
        for r in profiles.itertuples())
    drivers = ", ".join(f"{friendly(g, 'en')} ({s:.1f}% of gain)" for g, s in
                        zip(imp["group"].head(8), imp["gain_share"].head(8)))
    gloss = catalog("en")
    glossary = "\n".join(f"- {gloss[f'glossary.{k}.term']}: {gloss[f'glossary.{k}.def']}"
                         for k in ("churn", "probability", "roc_auc", "pp", "segment", "reason_code", "decile",
                                   "lift", "psi", "exploratory_band"))
    return f"""You are Churn Copilot, the assistant inside the Expresso Churn Intelligence Hub, a decision-support app \
built by Team Spartans for Expresso Senegal (a mobile operator). Your users are managers, the retention/CRM team and \
analysts.

Rules:
- Answer only from the facts below and from tool results. If neither contains what is needed, say "I don't have that \
data" and suggest which page or file would provide it. Never invent, estimate or round away figures; quote numbers as \
the tools return them.
- Keep a decision-support tone: probabilities are not verdicts, risk thresholds are not approved, recommended actions \
are suggestions for people to decide on.
- When a tool result says synthetic_demo_data is true, say clearly that the batch is synthetic demo data.
- Call tools whenever the question is about the session's scored batch, a what-if change, or details not listed here.
- Be concise: short paragraphs or bullets, plain business language, no code.
- Always answer in {LANG_NAMES.get(lang, "English")}.

Model facts (model_metadata.json):
- {meta['model_name']} classifier, validation ROC-AUC {meta['validation_roc_auc']:.4f} (seed {meta['validation_split_seed']}), \
{meta['training_rows']:,} training rows, {meta['feature_count']} engineered features from 17 raw columns.
- Risk thresholds approved: {meta['risk_thresholds_approved']}. The app shows probabilities; any bands are exploratory.
- Top drivers by gain: {drivers}.

Portfolio (segment_profiles.csv): {pf.total_customers:,} customers, overall churn {pf.churn_rate:.2f}%, about \
{pf.est_churners:,.0f} churners.
{seg_lines}

Glossary:
{glossary}

App pages: Home, The Problem, Executive Dashboard, Data Explorer, Customer Journey, Batch Scoring, \
Data Quality & Drift, Retention Planner, Revenue at Risk, Segment Explorer, Experiment Designer, Campaign Tracker, \
Methodology, Model & Explainability, Model Evaluation, Reports."""


def client_from_secrets(get_secret) -> tuple[anthropic.Anthropic | None, str, str]:
    key = get_secret("ANTHROPIC_API_KEY")
    model = get_secret("COPILOT_MODEL") or DEFAULT_MODEL
    effort = get_secret("COPILOT_EFFORT") or DEFAULT_EFFORT
    if not key:
        return None, model, effort
    import anthropic

    return anthropic.Anthropic(api_key=str(key)), str(model), str(effort)


def stream_answer(client: anthropic.Anthropic, model: str, effort: str, lang: str, messages: list,
                  ctx: T.Context, sources: list) -> Iterator[str]:
    """Yield answer text as it streams; run tools in between.

    ``messages`` is the API history (mutated in place: assistant turns are
    appended with their full content so thinking blocks are preserved).
    ``sources`` collects (tool name, arguments, is_error) for the UI.
    """
    # The language instruction lives in the (stable) system prompt: the history
    # is append-only, so preserved thinking blocks stay valid across turns.
    system = [{"type": "text", "text": system_prompt(lang), "cache_control": {"type": "ephemeral"}}]
    use_fallbacks = model in FALLBACK_MODELS
    json_retries = 0
    turn = 0
    while turn < MAX_TOOL_TURNS:
        kwargs = dict(model=model, max_tokens=16000, system=system, tools=T.api_schemas(),
                      messages=messages, output_config={"effort": effort})
        try:
            if use_fallbacks:
                cm = client.beta.messages.stream(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            else:
                cm = client.messages.stream(**kwargs)
            with cm as stream:
                for event in stream:
                    if event.type == "text":
                        yield event.text
                response = stream.get_final_message()
            json_retries = 0
        except ValueError:
            # Tool input JSON the SDK could not parse at all: re-issue the turn (bounded).
            json_retries += 1
            if json_retries > 2:
                raise
            continue
        turn += 1
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            yield "\n\n_" + _t(lang, "copilot.refusal") + "_"
            return
        if response.stop_reason == "pause_turn":
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return
        if response.stop_reason == "max_tokens":
            yield "\n\n_" + _t(lang, "copilot.truncated") + "_"
            return
        results = []
        for block in tool_uses:
            result, is_error = T.call(block.name, block.input, ctx)
            sources.append((block.name, block.input if isinstance(block.input, dict) else {}, is_error))
            results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": is_error,
                            "content": json.dumps(result, ensure_ascii=False, default=str)})
        messages.append({"role": "user", "content": results})
        yield "\n\n"
    yield "\n\n_" + _t(lang, "copilot.turn_limit") + "_"


def _t(lang: str, key: str) -> str:
    return catalog(lang).get(key) or catalog("en").get(key, key)
