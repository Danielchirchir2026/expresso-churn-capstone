"""Measure time and peak memory for scoring a large CSV, the way the app does.

    python scripts/benchmark_batch.py --rows 100000

Generates synthetic rows, writes them to a temporary CSV, reads it back,
validates, cleans and scores in chunks with reason codes and segments.
Peak memory is the process's peak resident set size (needs psutil).
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from app.core import demo, scoring, validation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=scoring.DEFAULT_CHUNK)
    ap.add_argument("--explain", type=int, default=scoring.DEFAULT_REASONS_FOR,
                    help="reason codes for the riskiest N rows (-1 = all rows)")
    args = ap.parse_args()

    try:
        import psutil
    except ImportError:  # pragma: no cover
        psutil = None
    proc = psutil.Process() if psutil else None
    peak = {"rss": 0}
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            if proc:
                peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            time.sleep(0.05)

    threading.Thread(target=sample, daemon=True).start()
    base = proc.memory_info().rss if proc else 0

    bundle = joblib.load(ROOT / "artifacts" / "expresso_churn_bundle.joblib")
    config = bundle["preprocessing_config"]
    profiles = pd.read_csv(ROOT / "artifacts" / "segment_profiles.csv")
    after_model = proc.memory_info().rss if proc else 0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "batch.csv"
        demo.generate(args.rows, profiles, config, seed=1).to_csv(path, index=False)
        size_mb = path.stat().st_size / 1e6
        data = path.read_bytes()  # the app receives uploads as bytes
        t0 = time.perf_counter()
        raw = pd.read_csv(io.BytesIO(data), low_memory=False)
        report = validation.validate(raw, config)
        clean = validation.clean(raw, config)
        t1 = time.perf_counter()
        scored = scoring.score_batch(clean, bundle, chunk_size=args.chunk,
                                     reasons_for=None if args.explain < 0 else args.explain)
        t2 = time.perf_counter()
    stop.set()

    mb = 1024 ** 2
    print(f"rows scored          : {len(scored):,}  (csv {size_mb:.1f} MB, blocking issues: {len(report.blocking)})")
    print(f"read + validate      : {t1 - t0:.1f} s")
    print(f"score (chunk {args.chunk:,}) : {t2 - t1:.1f} s  (reason codes for {int(scored.get('reasons_computed', scored['churn_probability'] * 0).sum()):,} rows)")
    if proc:
        print(f"RSS at start         : {base / mb:.0f} MB")
        print(f"RSS after model load : {after_model / mb:.0f} MB")
        print(f"peak RSS             : {peak['rss'] / mb:.0f} MB")
    else:
        print("install psutil to report memory")


if __name__ == "__main__":
    main()
