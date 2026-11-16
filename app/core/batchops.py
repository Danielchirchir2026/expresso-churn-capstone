"""Score a cleaned batch and keep it in session state for other pages."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from app.core import artifacts, demo, scoring, state


def score_and_store(clean_df: pd.DataFrame, name: str, synthetic: bool,
                    progress: Callable[[float], None] | None = None,
                    reasons_for: int | None = scoring.DEFAULT_REASONS_FOR) -> pd.DataFrame:
    bundle = artifacts.get_bundle()
    config = bundle["preprocessing_config"]
    scored = scoring.score_batch(clean_df, bundle, progress=progress, reasons_for=reasons_for)
    id_col = config["id_column"]
    extra = [c for c in clean_df.columns if c != id_col]
    merged = scored.merge(clean_df[[id_col] + extra], on=id_col, how="left", validate="one_to_one")
    drift = None
    try:  # drift report is optional (needs reference_stats.json)
        from app.core import drift as drift_mod

        ref = artifacts.get_optional("reference_stats")
        if ref:
            drift = drift_mod.report(clean_df, ref, config)
    except ImportError:
        pass
    state.set_batch(merged, name=name, synthetic=synthetic, drift=drift)
    return merged


def quick_demo(n: int = 5000, seed: int = 42, progress=None) -> pd.DataFrame:
    """Generate and score a synthetic demo batch in one step."""
    config = artifacts.get_config()
    df = demo.generate(n, artifacts.get_segment_profiles(), config, seed=seed,
                       breakdowns=artifacts.get_optional("eda_breakdowns"))
    return score_and_store(df, name=f"synthetic-demo-{n}", synthetic=True, progress=progress)
