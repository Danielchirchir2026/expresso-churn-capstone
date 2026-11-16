"""Export the small aggregate artifacts the app's evidence pages need.

Run once, locally, against the Zindi Train.csv:

    python scripts/export_artifacts.py --train path/to/Train.csv

Writes into artifacts/ (or --out):

    eda_summary.json               counts, churn rate, missingness, numeric stats by churn,
                                   per-segment medians (for Customer Check presets)
    eda_breakdowns.parquet         churn rate and counts by tenure, region, MRG,
                                   REGULARITY bins, top packs and value deciles
    validation_predictions.parquet y_true, y_prob, REGION, TENURE for the validation split (no IDs)
    model_comparison.csv           Dummy, Logistic Regression and the deployed LightGBM
    reference_stats.json           training distributions for drift (PSI) checks
    export_manifest.json           what was produced, split settings, sanity checks

Everything except validation_predictions.parquet is an aggregate; that file holds
labels, probabilities, region and tenure only. The script reuses the app's own
preprocessing (app/utils/prediction.py) and never modifies the model.

Split: the training notebook used seed 42 (model_metadata.json). Pass the same
--test-size and stratification the notebook used (defaults: 0.2, stratified).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.core.segments import assign_segments  # noqa: E402
from app.utils.prediction import prepare_customer_features  # noqa: E402

NUMERIC = ["MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE", "DATA_VOLUME", "ON_NET", "ORANGE",
           "TIGO", "ZONE1", "ZONE2", "REGULARITY", "FREQ_TOP_PACK"]
CATEGORICAL = ["REGION", "TENURE", "MRG", "TOP_PACK"]
DECILE_COLS = ["MONTANT", "REVENUE", "DATA_VOLUME", "ON_NET"]
CHUNK = 200_000


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(s: pd.Series) -> dict:
    v = s.dropna()
    if v.empty:
        return {"count": 0}
    q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {"count": int(v.size), "min": float(v.min()), "p01": float(q[0.01]), "q1": float(q[0.25]),
            "median": float(q[0.5]), "q3": float(q[0.75]), "p99": float(q[0.99]), "max": float(v.max()),
            "mean": float(v.mean())}


def churn_table(key: pd.Series, y: pd.Series, dimension: str, order=None) -> pd.DataFrame:
    g = pd.DataFrame({"level": key.astype("string").fillna("Not recorded"), "y": y}).groupby("level")["y"]
    out = pd.DataFrame({"count": g.size(), "churners": g.sum()}).reset_index()
    out["churn_rate"] = out["churners"] / out["count"]
    out["dimension"] = dimension
    if order is not None:
        rank = {lvl: i for i, lvl in enumerate(order)}
        out["order"] = out["level"].map(lambda x: rank.get(x, len(rank)))
    else:
        out["order"] = out["count"].rank(ascending=False, method="first").astype(int)
    return out


def decile_table(values: pd.Series, y: pd.Series, dimension: str) -> pd.DataFrame:
    v = pd.to_numeric(values, errors="coerce")
    present = v.notna()
    labels = pd.Series(pd.NA, index=v.index, dtype="string")
    if present.any():
        codes = pd.qcut(v[present].rank(method="first"), 10, labels=False)
        bounds = v[present].groupby(codes).agg(["min", "max"])
        names = {int(i): f"D{int(i) + 1} ({r['min']:,.0f}-{r['max']:,.0f})" for i, r in bounds.iterrows()}
        labels[present] = codes.map(names).astype("string")
    order = [f"D{i}" for i in range(1, 11)]
    t = churn_table(labels, y, dimension)
    t["order"] = t["level"].map(lambda s: int(s[1:s.index(" ")]) if s.startswith("D") else 11)
    return t


def reference_numeric(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce")
    present = v.dropna()
    edges = np.unique(np.quantile(present, np.linspace(0, 1, 11))) if len(present) else np.array([])
    inner = edges[1:-1]
    counts = np.bincount(np.searchsorted(inner, present.to_numpy(), side="right"), minlength=len(inner) + 1)
    return {"edges": [float(e) for e in inner], "proportions": [float(c) / max(len(v), 1) for c in counts],
            "missing_rate": float(v.isna().mean())}


def reference_categorical(s: pd.Series, top: int | None = None) -> dict:
    v = s.astype("string").str.strip()
    counts = v.value_counts(dropna=True)
    if top is not None and len(counts) > top:
        other = counts.iloc[top:].sum()
        counts = counts.iloc[:top]
        counts.loc["__other__"] = other
    return {"proportions": {str(k): float(c) / len(v) for k, c in counts.items()},
            "missing_rate": float(v.isna().mean())}


def score(model, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.booster_.predict(features), dtype="float64")


def features_in_chunks(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    parts = [prepare_customer_features(df.iloc[i:i + CHUNK], config) for i in range(0, len(df), CHUNK)]
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="path to Zindi Train.csv")
    ap.add_argument("--out", default=str(ROOT / "artifacts"), help="output folder (default: artifacts/)")
    ap.add_argument("--test-size", type=float, default=0.2, help="validation share used in training (default 0.2)")
    ap.add_argument("--no-stratify", action="store_true", help="the notebook split without stratification")
    ap.add_argument("--baseline-sample", type=int, default=400_000,
                    help="training rows used to fit the baseline models (default 400,000; 0 = all)")
    ap.add_argument("--holdout-refit", action="store_true",
                    help="also train a LightGBM with the deployed settings on the training split only and use it "
                         "for validation predictions (honest evaluation if the deployed model saw all rows)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(ROOT / "artifacts" / "expresso_churn_bundle.joblib")
    config, model = bundle["preprocessing_config"], bundle["model"]
    meta = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
    seed = int(meta.get("validation_split_seed", config.get("seed", 42)))
    id_col, target = config["id_column"], config.get("target_column", "CHURN")

    log(f"Reading {args.train}")
    dtypes = {c: "string" for c in [id_col] + CATEGORICAL}
    dtypes.update({c: "float64" for c in NUMERIC})
    df = pd.read_csv(args.train, dtype=dtypes)
    missing = [c for c in [id_col, target] + list(config["raw_source_columns"]) if c not in df.columns]
    if missing:
        sys.exit(f"Train.csv is missing columns: {missing}")
    y = df[target].astype("int8")
    log(f"{len(df):,} rows, churn rate {y.mean():.4%}")

    # ------------------------------------------------------------------ split
    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=seed,
                                      stratify=None if args.no_stratify else y)
    train, valid = df.iloc[tr_idx], df.iloc[va_idx]
    log(f"Split (seed {seed}, test_size {args.test_size}, stratified {not args.no_stratify}): "
        f"{len(train):,} train / {len(valid):,} validation")

    # ------------------------------------------------------------ EDA summary
    log("EDA summary")
    summary = {
        "source": Path(args.train).name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "churners": int(y.sum()), "churn_rate": float(y.mean()),
        "missing_pct": {c: float(df[c].isna().mean() * 100) for c in df.columns if c != id_col},
        "numeric": {c: {"all": stats(df[c]), "churn_0": stats(df.loc[y == 0, c]), "churn_1": stats(df.loc[y == 1, c])}
                    for c in NUMERIC},
    }

    # --------------------------------------------- segments: check + medians
    log("Assigning segments with the bundle's segmentation model (all rows, chunked)")
    seg = np.empty(len(df), dtype="int8")
    for i in range(0, len(df), CHUNK):
        part = df.iloc[i:i + CHUNK]
        seg[i:i + CHUNK] = assign_segments(part, prepare_customer_features(part, config), bundle)
    profiles = pd.read_csv(ROOT / "artifacts" / "segment_profiles.csv").set_index("segment")
    counts = pd.Series(seg).value_counts().reindex(profiles.index, fill_value=0)
    seg_rates = pd.Series(y.to_numpy()).groupby(seg).mean().reindex(profiles.index) * 100
    check = {int(s): {"expected_customers": int(profiles.loc[s, "customers"]), "reconstructed": int(counts[s]),
                      "expected_churn_rate": float(profiles.loc[s, "churn_rate"]),
                      "reconstructed_churn_rate": float(seg_rates[s])} for s in profiles.index}
    total = max(int(counts.sum()), 1)
    for s_id, v in check.items():
        v["expected_pct"] = float(profiles.loc[s_id, "pct_customers"])
        v["reconstructed_pct"] = 100.0 * v["reconstructed"] / total
    seg_ok = all(abs(v["reconstructed_pct"] - v["expected_pct"]) <= 0.5
                 and abs(v["reconstructed_churn_rate"] - v["expected_churn_rate"]) <= 0.5 for v in check.values())
    log(f"Segment reconstruction {'matches' if seg_ok else 'does NOT match'} segment_profiles.csv "
        "(shares and churn rates within 0.5 pp)")
    seg_medians, seg_modes = {}, {}
    for s in profiles.index:
        part = df[seg == s]
        seg_medians[str(int(s))] = {c: (None if part[c].isna().mean() > 0.5 else float(part[c].median()))
                                    for c in NUMERIC}
        seg_modes[str(int(s))] = {c: (None if part[c].isna().mean() > 0.5 or part[c].dropna().empty
                                      else str(part[c].mode().iloc[0])) for c in ("TENURE", "REGION", "TOP_PACK")}
    summary.update({"segment_medians": seg_medians, "segment_modes": seg_modes, "segment_check": check,
                    "segment_check_ok": bool(seg_ok)})

    # -------------------------------------------------------- EDA breakdowns
    log("EDA breakdowns")
    reg = pd.to_numeric(df["REGULARITY"], errors="coerce")
    bins = pd.cut(reg, bins=list(range(0, 91, 10)), labels=[f"{i * 10 + 1}-{i * 10 + 10}" for i in range(9)])
    top_packs = df["TOP_PACK"].value_counts().head(15).index
    pack = df["TOP_PACK"].where(df["TOP_PACK"].isin(top_packs) | df["TOP_PACK"].isna(), "Other packs")
    tables = [
        churn_table(df["TENURE"], y, "TENURE", order=config["tenure_order"]),
        churn_table(df["REGION"], y, "REGION"),
        churn_table(df["MRG"], y, "MRG"),
        churn_table(bins.astype("string"), y, "REGULARITY", order=[f"{i * 10 + 1}-{i * 10 + 10}" for i in range(9)]),
        churn_table(pack, y, "TOP_PACK"),
    ] + [decile_table(df[c], y, f"{c}_DECILE") for c in DECILE_COLS]
    breakdowns = pd.concat(tables, ignore_index=True)[["dimension", "level", "order", "count", "churners",
                                                       "churn_rate"]]
    breakdowns["count"] = breakdowns["count"].astype("int64")
    breakdowns["churners"] = breakdowns["churners"].astype("int64")

    # ---------------------------------------------------- reference (drift)
    log("Reference distributions (training split)")
    reference = {
        "source": f"training split of {Path(args.train).name} (seed {seed})", "rows": int(len(train)),
        "numeric": {c: reference_numeric(train[c]) for c in NUMERIC},
        "categorical": {"REGION": reference_categorical(train["REGION"]),
                        "TENURE": reference_categorical(train["TENURE"]),
                        "MRG": reference_categorical(train["MRG"]),
                        "TOP_PACK": reference_categorical(train["TOP_PACK"], top=30)},
    }

    # --------------------------------------------- validation predictions
    log("Scoring the validation split with the deployed model")
    X_va = features_in_chunks(valid, config)
    y_va = y.iloc[va_idx].to_numpy()
    p_deployed = score(model, X_va)
    auc_deployed = roc_auc_score(y_va, p_deployed)
    log(f"Deployed model AUC on this split: {auc_deployed:.4f} (recorded at training: {meta['validation_roc_auc']:.4f})")
    optimistic = auc_deployed > meta["validation_roc_auc"] + 0.002
    if optimistic:
        log("WARNING: AUC is above the recorded value: the deployed model probably saw these rows (final refit)."
            + ("" if args.holdout_refit else " Re-run with --holdout-refit for honest evaluation curves."))

    sample_n = len(train) if args.baseline_sample <= 0 else min(args.baseline_sample, len(train))
    rng = np.random.default_rng(seed)
    fit_idx = np.sort(rng.choice(len(train), size=sample_n, replace=False))
    X_fit = features_in_chunks(train.iloc[fit_idx], config)
    y_fit = y.iloc[tr_idx].to_numpy()[fit_idx]

    rows = []

    def add_row(name: str, prob: np.ndarray, seconds, note: str) -> None:
        rows.append({"model": name, "roc_auc": roc_auc_score(y_va, prob), "pr_auc": average_precision_score(y_va, prob),
                     "log_loss": log_loss(y_va, np.clip(prob, 1e-7, 1 - 1e-7)),
                     "f1_at_0_5": f1_score(y_va, prob >= 0.5), "train_seconds": seconds, "note": note})

    log("Baseline: Dummy (predicts the training churn rate)")
    t0 = time.perf_counter()
    dummy = DummyClassifier(strategy="prior").fit(X_fit, y_fit)
    add_row("Dummy (prior)", dummy.predict_proba(X_va)[:, 1], time.perf_counter() - t0,
            f"fit on {sample_n:,} training rows")

    log(f"Baseline: Logistic Regression on {sample_n:,} training rows")
    t0 = time.perf_counter()
    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=1000)).fit(X_fit, y_fit)
    add_row("Logistic Regression", logreg.predict_proba(X_va)[:, 1], time.perf_counter() - t0,
            f"median imputation + scaling, fit on {sample_n:,} training rows")

    add_row("LightGBM (deployed)", p_deployed, None,
            "saved bundle; training time not recorded" + ("; scores optimistic (saw validation rows)" if optimistic else ""))

    validation_model, p_valid = "deployed", p_deployed
    if args.holdout_refit:
        import lightgbm as lgb

        log("Holdout refit: LightGBM with the deployed settings on the full training split")
        params = model.get_params()
        params.update(n_estimators=int(meta["best_iteration"]), n_jobs=-1)
        t0 = time.perf_counter()
        X_tr = features_in_chunks(train, config)
        refit = lgb.LGBMClassifier(**params).fit(X_tr, y.iloc[tr_idx].to_numpy())
        p_refit = refit.predict_proba(X_va)[:, 1]
        add_row("LightGBM (holdout refit, deployed settings)", p_refit, time.perf_counter() - t0,
                f"n_estimators = best iteration {meta['best_iteration']}; trained on the training split only")
        validation_model, p_valid = "holdout_refit", p_refit
        del X_tr

    comparison = pd.DataFrame(rows)
    preds = pd.DataFrame({"y_true": y_va.astype("int8"), "y_prob": p_valid.astype("float32"),
                          "REGION": valid["REGION"].astype("string").to_numpy(),
                          "TENURE": valid["TENURE"].astype("string").to_numpy()})

    # ------------------------------------------------------------- write
    files = {
        "eda_summary.json": lambda p: p.write_text(json.dumps(summary, indent=1), encoding="utf-8"),
        "eda_breakdowns.parquet": lambda p: breakdowns.to_parquet(p, index=False),
        "validation_predictions.parquet": lambda p: preds.to_parquet(p, index=False, compression="zstd"),
        "model_comparison.csv": lambda p: comparison.to_csv(p, index=False),
        "reference_stats.json": lambda p: p.write_text(json.dumps(reference, indent=1), encoding="utf-8"),
    }
    sizes = {}
    for name, write in files.items():
        path = out / name
        write(path)
        sizes[name] = path.stat().st_size
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": Path(args.train).name, "rows": int(len(df)),
        "split": {"seed": seed, "test_size": args.test_size, "stratified": not args.no_stratify,
                  "train_rows": int(len(train)), "validation_rows": int(len(valid))},
        "validation_model": validation_model,
        "deployed_auc_on_split": float(auc_deployed), "recorded_auc": float(meta["validation_roc_auc"]),
        "deployed_scores_optimistic": bool(optimistic),
        "segment_check_ok": bool(seg_ok),
        "baseline_training_rows": int(sample_n),
        "files": sizes, "total_mb": round(sum(sizes.values()) / 1e6, 2),
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"Wrote {len(files) + 1} files to {out} ({manifest['total_mb']} MB) in {manifest['seconds']} s")
    for name, size in sizes.items():
        log(f"  {name:32s} {size / 1e6:8.2f} MB")
    if manifest["total_mb"] > 20:
        log("WARNING: total size is above 20 MB; consider a smaller --test-size before committing.")


if __name__ == "__main__":
    main()
