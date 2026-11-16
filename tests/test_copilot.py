"""Copilot tools, offline FAQ and the streaming tool loop (with a fake client)."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app.copilot import faq, llm
from app.copilot import tools as T
from app.core import scoring


@pytest.fixture(scope="module")
def ctx_with_batch(edge_rows, bundle):
    scored = scoring.score_batch(edge_rows, bundle)
    df = scored.merge(edge_rows, on="user_id")
    batch = {"df": df, "name": "edge", "synthetic": True, "rows": len(df)}
    return T.Context(batch=batch, thresholds=(0.2, 0.5), lang="en")


def test_every_tool_returns_json_safe_aggregates(ctx_with_batch):
    for name, args in [("portfolio_overview", {}), ("segment_profile", {"name": "silent"}), ("model_card", {}),
                       ("batch_summary", {}), ("top_customers", {"n": 3}), ("risk_by", {"column": "REGION"}),
                       ("run_what_if", {"base": "light_active", "changes": {"REGULARITY": 60}})]:
        result, err = T.call(name, args, ctx_with_batch)
        assert not err, (name, result)
        json.dumps(result)  # serialisable


def test_tool_input_validation(ctx_with_batch):
    assert T.call("top_customers", {"n": 500}, ctx_with_batch)[1]           # cap at 20
    assert T.call("run_what_if", {"base": "silent", "changes": {"REGION": 1}}, ctx_with_batch)[1]  # not a lever
    assert T.call("risk_by", {"column": "user_id"}, ctx_with_batch)[1]     # no raw rows by ID
    assert T.call("nope", {}, ctx_with_batch)[1]
    assert T.call("segment_profile", {"name": "silent", "extra": 1}, ctx_with_batch)[1]
    assert T.call("batch_summary", {}, T.Context())[1]                      # no batch -> error, not crash


def test_top_customers_never_returns_raw_rows(ctx_with_batch):
    result, _ = T.call("top_customers", {"n": 3}, ctx_with_batch)
    assert set(result["customers"][0]) == {"user_id", "probability_pct", "risk_decile", "segment",
                                           "top_reasons_raising_risk"}


@pytest.mark.parametrize("question,topic", [
    ("Which segment should we prioritise?", "prioritise"),
    ("Why is Silent / Inactive churning?", "silent"),
    ("Summarise my uploaded batch", "batch"),
    ("What does ROC-AUC 0.93 mean?", "auc"),
    ("Pourquoi les Silencieux / Inactifs partent-ils ?", "silent"),
    ("Quel segment faut-il prioriser ?", "prioritise"),
])
def test_faq_routing(question, topic):
    assert faq.match(question) == topic


def test_faq_answers_use_artifact_numbers(ctx_with_batch):
    text, sources = faq.answer("Why is Silent / Inactive churning?", T.Context(lang="en"))
    assert "53.1%" in text and sources
    text_fr, _ = faq.answer("Pourquoi les Silencieux / Inactifs partent-ils ?", T.Context(lang="fr"))
    assert "53,1" in text_fr
    text, _ = faq.answer("Summarise my uploaded batch", ctx_with_batch)
    assert "5 customers" in text and "Synthetic" in text


# ---------------------------------------------------------------------------
# Streaming tool loop with a fake Anthropic client
# ---------------------------------------------------------------------------
class FakeStream:
    def __init__(self, texts, final):
        self._events = [SimpleNamespace(type="text", text=x) for x in texts]
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class FakeMessages:
    def __init__(self, turns, log):
        self.turns, self.log = list(turns), log

    def stream(self, **kwargs):
        self.log.append(kwargs)
        texts, final = self.turns.pop(0)
        return FakeStream(texts, final)


def test_stream_loop_runs_tools_and_preserves_history():
    tool_turn = SimpleNamespace(stop_reason="tool_use", content=[
        SimpleNamespace(type="tool_use", id="tu_1", name="segment_profile", input={"name": "silent"})])
    text_turn = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Done.")])
    log: list = []
    beta = SimpleNamespace(messages=FakeMessages([(["Checking… "], tool_turn), (["Done."], text_turn)], log))
    client = SimpleNamespace(beta=beta, messages=None)
    history = [{"role": "user", "content": "Tell me about silent customers"}]
    sources: list = []
    out = "".join(llm.stream_answer(client, "claude-opus-5-5", "medium", "en", history, T.Context(), sources))

    assert "Checking" in out and out.endswith("Done.")
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[2]["content"][0]["type"] == "tool_result" and not history[2]["content"][0]["is_error"]
    assert sources == [("segment_profile", {"name": "silent"}, False)]
    first = log[0]
    assert first["fallbacks"] == "default" and first["betas"] == [llm.FALLBACK_BETA]
    assert first["output_config"] == {"effort": "medium"}
    assert all(t["eager_input_streaming"] for t in first["tools"])
    assert "Always answer in English" in first["system"][0]["text"]


def test_stream_loop_handles_refusal_without_fallback_model():
    refusal = SimpleNamespace(stop_reason="refusal", content=[])
    log: list = []
    client = SimpleNamespace(messages=FakeMessages([([], refusal)], log), beta=None)
    out = "".join(llm.stream_answer(client, "claude-haiku-5-5", "low", "fr", [{"role": "user", "content": "x"}],
                                    T.Context(lang="fr"), []))
    assert "Je ne peux pas" in out
    assert "fallbacks" not in log[0]


# ---------------------------------------------------------------------------
# Phase 14: tools over exported artifacts, summary, new FAQ topics
# ---------------------------------------------------------------------------
def test_extra_tools_without_exports_say_so():
    for name, args in [("model_comparison", {}), ("eda_breakdown", {"column": "TENURE"}),
                       ("evaluation_at_threshold", {"threshold": 0.5})]:
        result, err = T.call(name, args, T.Context())
        if err:  # repo artifacts/ has no exported files
            assert "I don't have that data" in result["error"]
    ss, err = T.call("sample_size", {"reduction": 20}, T.Context())
    assert not err and ss["customers_per_group"] > 0


def test_extra_tools_with_exports(exported, ctx_with_batch, monkeypatch):
    from app.core import artifacts

    monkeypatch.setattr(artifacts, "OPTIONAL_DIR", exported)
    for name, args in [("model_comparison", {}), ("eda_breakdown", {"column": "TENURE"}),
                       ("evaluation_at_threshold", {"threshold": 0.4}), ("revenue_at_risk", {}),
                       ("drift_report", {})]:
        result, err = T.call(name, args, ctx_with_batch)
        assert not err, (name, result)
        json.dumps(result)


def test_executive_summary_template_and_pdf(ctx_with_batch):
    from app.copilot import summary

    facts, sources = summary.facts(ctx_with_batch)
    assert {"portfolio", "model", "batch"} <= set(facts) and sources
    for lang in ("en", "fr"):
        text = summary.template_summary(facts, lang)
        assert "###" in text and ("18.75%" in text or "18,75" in text)
        pdf = summary.summary_pdf(text, lang)
        assert pdf[:5] == b"%PDF-"


@pytest.mark.parametrize("question,topic", [
    ("Is my batch drifting from the training data?", "drift"),
    ("How big should an A/B test be?", "experiment"),
    ("How does the model compare with the baseline?", "comparison"),
])
def test_faq_routing_new_topics(question, topic):
    assert faq.match(question) == topic


def test_llm_summary_streams_with_fake_client(ctx_with_batch):
    from app.copilot import summary

    facts, _ = summary.facts(ctx_with_batch)
    final = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="### Portfolio")])
    log: list = []
    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages([(["### Portfolio", " ok"], final)], log)),
                             messages=None)
    text = "".join(summary.stream_llm_summary(client, "claude-opus-5-5", "medium", "fr", facts))
    assert text == "### Portfolio ok"
    sent = log[0]
    assert sent["fallbacks"] == "default" and sent["output_config"] == {"effort": "medium"}
    assert "Write in French" in sent["system"] and "rule_based_recommendations" in sent["messages"][0]["content"]
