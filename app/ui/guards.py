"""Empty states for pages that depend on exported artifacts."""

from __future__ import annotations

from app.core import artifacts
from app.i18n import t
from app.ui.components import empty_state


def require_artifact(name: str, detail_key: str | None = None):
    """Return the optional artifact, or show an empty state naming the file and command."""
    data = artifacts.get_optional(name)
    if data is None:
        empty_state(t("ui.missing_artifact", file=f"artifacts/{artifacts.OPTIONAL_FILES[name]}"),
                    t(detail_key) if detail_key else t("ui.missing_artifact_detail"),
                    command=artifacts.EXPORT_COMMAND, icon="▤")
    return data
