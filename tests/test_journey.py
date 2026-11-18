"""Customer Journey page: one customer followed through all five steps."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.i18n import t

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["jr-find", "jr-risk", "jr-why", "jr-whatif", "jr-act"]


def open_journey() -> AppTest:
    at = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=300).run()
    at.switch_page("app_pages/journey.py").run()
    assert_clean(at)
    return at


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert t("ui.error.title", "en") not in [e.value for e in at.error]


def html_text(at: AppTest) -> str:
    return " ".join(h.value for h in at.get("html"))


def test_opens_on_a_scored_customer_with_every_step():
    at = open_journey()
    page = html_text(at)
    for anchor in STEPS:
        assert f'id="{anchor}"' in page, anchor          # each step header renders
        assert f'href="#{anchor}"' in page, anchor       # and the strip links to it
    assert at.session_state["cc_values"]["user_id"] == "CUSTOMER-001"
    assert at.session_state["last_customer"]["user_id"] == "CUSTOMER-001"  # handed to the Copilot


def test_preset_switch_resets_the_what_if_levers():
    at = open_journey()
    at.slider(key="wi_REGULARITY").set_value(90).run()
    assert_clean(at)
    at.button(key="jr_preset_4").click().run()
    assert_clean(at)
    origin = at.session_state["jr_origin"]
    assert origin["kind"] == "preset" and origin["segment"] == 4
    expected = int(round(at.session_state["cc_values"]["REGULARITY"]))
    assert at.slider(key="wi_REGULARITY").value == expected  # levers follow the new customer

    at.slider(key="wi_REGULARITY").set_value(90).run()
    at.button(key="jr_preset_4").click().run()  # reloading the same customer also restarts the levers
    assert_clean(at)
    assert at.slider(key="wi_REGULARITY").value == expected


def test_hand_entry_rescores_the_journey():
    at = open_journey()
    at.segmented_control(key="jr_source").set_value("manual").run()
    assert_clean(at)
    at.text_input(key="cc_user_id").set_value("AMINA-42")
    at.number_input(key="cc_REVENUE").set_value(25000.0)
    at.checkbox(key="cc_REVENUE_na").uncheck()  # the Silent default has no revenue recorded
    at.form_submit_button[0].click().run()
    assert_clean(at)
    assert at.session_state["cc_values"]["user_id"] == "AMINA-42"
    assert at.session_state["cc_values"]["REVENUE"] == 25000.0
    assert at.session_state["jr_origin"] == {"kind": "manual"}
    assert "AMINA-42" in html_text(at)


def test_follow_a_customer_from_a_demo_batch():
    at = open_journey()
    at.segmented_control(key="jr_source").set_value("batch").run()
    assert_clean(at)
    at.button(key="jr_quick_demo").click().run()  # no batch yet: one-click demo batch
    assert_clean(at)
    batch = at.session_state["batch"]
    riskiest = batch["df"].nlargest(50, "churn_probability")["user_id"].astype(str).tolist()
    chosen = riskiest[1]
    at.selectbox(key=f"jr_batch_pick_{batch['id']}").select_index(1)
    at.button(key="jr_batch_go").click().run()
    assert_clean(at)
    assert at.session_state["cc_values"]["user_id"] == chosen
    assert at.session_state["jr_origin"]["kind"] == "batch"

    at.text_input(key="jr_batch_search").set_value("no-such-customer")
    at.button(key="jr_batch_go").click().run()
    assert_clean(at)
    assert at.session_state["cc_values"]["user_id"] == chosen  # unchanged
    assert any("no-such-customer" in w.value for w in at.warning)


def test_old_pages_are_gone_from_navigation():
    from app.navigation import PAGES

    assert "customer_check" not in PAGES and "what_if" not in PAGES
    assert PAGES["journey"][2] == "predict" and PAGES["drift"][2] == "trust"
