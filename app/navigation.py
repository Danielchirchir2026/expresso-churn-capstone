"""Page registry, navigation sections and role-based ordering."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.i18n import t

ROOT = Path(__file__).resolve().parents[1]

# key -> (page file, material icon, section)
PAGES: dict[str, tuple[str, str, str]] = {
    "home": ("app_pages/home.py", ":material/home:", ""),
    "problem": ("app_pages/problem.py", ":material/flag:", "overview"),
    "executive": ("app_pages/executive.py", ":material/monitoring:", "overview"),
    "data_explorer": ("app_pages/data_explorer.py", ":material/query_stats:", "overview"),
    "customer_check": ("app_pages/customer_check.py", ":material/person_search:", "predict"),
    "what_if": ("app_pages/what_if.py", ":material/tune:", "predict"),
    "batch": ("app_pages/batch.py", ":material/upload_file:", "predict"),
    "drift": ("app_pages/drift.py", ":material/troubleshoot:", "predict"),
    "retention": ("app_pages/retention.py", ":material/savings:", "act"),
    "revenue_risk": ("app_pages/revenue_risk.py", ":material/payments:", "act"),
    "segments": ("app_pages/segments.py", ":material/groups:", "act"),
    "experiment": ("app_pages/experiment.py", ":material/science:", "act"),
    "campaigns": ("app_pages/campaigns.py", ":material/campaign:", "act"),
    "methodology": ("app_pages/methodology.py", ":material/schema:", "trust"),
    "model": ("app_pages/model.py", ":material/model_training:", "trust"),
    "evaluation": ("app_pages/evaluation.py", ":material/fact_check:", "trust"),
    "copilot": ("app_pages/copilot.py", ":material/smart_toy:", "assistant"),
    "reports": ("app_pages/reports.py", ":material/summarize:", "reports"),
}

SECTION_ORDER = {
    "all": ["overview", "predict", "act", "trust", "assistant", "reports"],
    "executive": ["overview", "act", "assistant", "reports", "predict", "trust"],
    "retention": ["act", "predict", "assistant", "overview", "reports", "trust"],
    "analyst": ["trust", "overview", "predict", "act", "assistant", "reports"],
}

# Pages each role sees first inside their sections.
ROLE_PRIORITY = {
    "all": [],
    "executive": ["executive", "problem", "revenue_risk", "retention", "segments", "campaigns"],
    "retention": ["retention", "campaigns", "experiment", "segments", "batch", "customer_check", "what_if"],
    "analyst": ["model", "evaluation", "methodology", "data_explorer", "drift", "batch"],
}
ROLES = ["all", "executive", "retention", "analyst"]


def page_path(key: str) -> str | None:
    """Relative path of a page that exists (None if not built yet)."""
    rel = PAGES[key][0]
    return rel if (ROOT / rel).exists() else None


def available(key: str) -> bool:
    return page_path(key) is not None


def build(role: str = "all"):
    """Build the st.navigation object for the chosen role view."""
    role = role if role in SECTION_ORDER else "all"
    priority = ROLE_PRIORITY[role]
    default_order = list(PAGES)

    def rank(key: str) -> tuple[int, int]:
        return (priority.index(key) if key in priority else len(priority), default_order.index(key))

    sections: dict[str, list] = {"": []}
    for section in SECTION_ORDER[role]:
        sections[t(f"nav.section.{section}")] = []
    section_labels = {s: t(f"nav.section.{s}") for s in SECTION_ORDER["all"]}

    for key in sorted(PAGES, key=rank):
        path = page_path(key)
        if path is None:
            continue
        _, icon, section = PAGES[key]
        page = st.Page(path, title=t(f"nav.page.{key}"), icon=icon, url_path=key.replace("_", "-"),
                       default=(key == "home"))
        label = "" if section == "" else section_labels[section]
        sections[label].append(page)

    return st.navigation({k: v for k, v in sections.items() if v}, position="sidebar", expanded=True)


def goto(key: str) -> None:
    path = page_path(key)
    if path:
        st.switch_page(path)


def link(key: str, label: str | None = None, icon: str | None = None, **kwargs) -> None:
    """st.page_link to a registered page, silently skipped if not built yet."""
    path = page_path(key)
    if path:
        st.page_link(path, label=label or t(f"nav.page.{key}"), icon=icon or PAGES[key][1], **kwargs)
