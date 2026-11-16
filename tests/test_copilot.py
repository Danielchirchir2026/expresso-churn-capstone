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
