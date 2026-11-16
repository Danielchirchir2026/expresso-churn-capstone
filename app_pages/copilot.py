"""Churn Copilot: ask questions in plain language, answered from the app's data."""

import json

import anthropic
import streamlit as st

from app.copilot import faq, llm
from app.copilot import tools as T
from app.core import artifacts, fmt, state
from app.i18n import current_lang, t
from app.ui import layout
from app.ui.components import badge, page_header

lang = current_lang()
client, model, effort = llm.client_from_secrets(layout.get_secret)
meta = artifacts.get_metadata()

page_header(t("copilot.title"), t("copilot.sub"), t("copilot.eyebrow"))
mode_badge = (badge(t("copilot.mode.online", model=model), "ok") if client
              else badge(t("copilot.mode.offline"), "warn"))
batch = state.get_batch()
data_badge = badge(t("sidebar.batch_loaded", rows=fmt.num(batch["rows"])), "ok") if batch else badge(
    t("sidebar.no_batch"), "muted")
st.html(f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin:-.2rem 0 .6rem 0">{mode_badge}{data_badge}</div>')
if not client:
    st.caption(t("copilot.mode.offline_help"))

# ---------------------------------------------------------------------------
# Conversation state (display history + API history kept separately)
# ---------------------------------------------------------------------------
st.session_state.setdefault("cp_display", [])
st.session_state.setdefault("cp_api", [])
if st.session_state.get("cp_lang") != lang:  # new language -> new API conversation
    st.session_state["cp_api"] = []
    st.session_state["cp_lang"] = lang


def render_sources(sources: list) -> None:
    with st.expander(t("copilot.sources"), icon=":material/fact_check:"):
        st.markdown(f"- {t('copilot.grounding')}")
        for name, args, is_error in sources:
            label = t(f"copilot.tool.{name}")
            arg_txt = f" `{json.dumps(args, ensure_ascii=False)}`" if args else ""
            flag = f" ({t('copilot.tool_error')})" if is_error else ""
            st.markdown(f"- {label}{arg_txt}{flag}")


STARTERS = ["prioritise", "silent", "batch", "auc"]


def pick_starter() -> None:
    choice = st.session_state.get("cp_chip")
    if choice:
        st.session_state["cp_pending"] = t(f"copilot.starter.{choice}", auc=fmt.num(meta["validation_roc_auc"], 2))
    st.session_state["cp_chip"] = None


st.pills(t("copilot.starters"), STARTERS, key="cp_chip", on_change=pick_starter,
         format_func=lambda k: t(f"copilot.starter.{k}", auc=fmt.num(meta["validation_roc_auc"], 2)))
if st.session_state["cp_display"] and st.button(t("copilot.clear"), icon=":material/delete_sweep:", key="cp_clear"):
    st.session_state["cp_display"] = []
    st.session_state["cp_api"] = []

if not st.session_state["cp_display"]:
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.markdown(t("copilot.welcome"))

for msg in st.session_state["cp_display"]:
    with st.chat_message(msg["role"], avatar=":material/smart_toy:" if msg["role"] == "assistant" else None):
        st.markdown(msg["text"])
        if msg["role"] == "assistant" and msg.get("sources") is not None:
            render_sources(msg["sources"])

prompt = st.chat_input(t("copilot.input"))
prompt = st.session_state.pop("cp_pending", None) or prompt

if prompt:
    st.session_state["cp_display"].append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    ctx = T.Context(batch=state.get_batch(), last_customer=state.get_last_customer(),
                    thresholds=st.session_state.get("band_thresholds"), lang=lang)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        sources: list = []
        text = None
        if client:
            api = st.session_state["cp_api"]
            api.append({"role": "user", "content": prompt})
            try:
                text = st.write_stream(llm.stream_answer(client, model, effort, lang, api, ctx, sources))
            except anthropic.AuthenticationError:
                st.warning(t("copilot.err.auth"), icon=":material/key_off:")
            except anthropic.RateLimitError:
                st.warning(t("copilot.err.rate"), icon=":material/hourglass_top:")
            except anthropic.APIStatusError as exc:
                st.warning(t("copilot.err.status", code=exc.status_code), icon=":material/cloud_off:")
            except anthropic.APIConnectionError:
                st.warning(t("copilot.err.connection"), icon=":material/wifi_off:")
            if text is None:  # the API turn failed: start a fresh API conversation, answer offline
                st.session_state["cp_api"] = []
                sources = []
        if text is None:
            text, sources = faq.answer(prompt, ctx)
            st.markdown(text)
        render_sources(sources)
    st.session_state["cp_display"].append({"role": "assistant", "text": text if isinstance(text, str) else str(text),
                                           "sources": sources})

st.caption(t("copilot.privacy"))
