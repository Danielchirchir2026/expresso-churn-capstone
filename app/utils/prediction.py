from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
BUNDLE_PATH = PROJECT_DIR / "artifacts" / "expresso_churn_bundle.joblib"


def load_bundle():
    """Load the saved Expresso churn deployment bundle."""
    return joblib.load(BUNDLE_PATH)


def prepare_customer_features(raw_df, config):
    """
    Validate, clean and transform raw customer rows
    using the preprocessing rules saved during model training.
    """

    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame.")

    id_col = config["id_column"]
    required_columns = [id_col] + config["raw_source_columns"]

    missing_columns = [
        column
        for column in required_columns
        if column not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input data is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Keep only expected source columns
    source = raw_df[required_columns].copy()

    if source[id_col].isna().any():
        raise ValueError("Customer IDs contain missing values.")

    if source[id_col].duplicated().any():
        raise ValueError(
            "Customer IDs must be unique within the prediction batch."
        )

    # Match categorical cleaning used during model development
    for column in config["categorical_columns"]:
        source[column] = (
            source[column]
            .astype("string")
            .str.strip()
            .fillna("MISSING")
        )

    # Recreate missing-value indicators
    for column in config["columns_with_missing_values"]:
        source[f"{column}_MISSING"] = (
            raw_df[column].isna().astype("int8")
        )

    original_predictors = config["raw_source_columns"]

    source["MISSING_COUNT"] = (
        raw_df[original_predictors]
        .isna()
        .sum(axis=1)
        .astype("int8")
    )

    # Build model features
    out = pd.DataFrame(index=source.index)

    for column in config["numeric_columns"]:
        out[column] = pd.to_numeric(
            source[column],
            errors="coerce"
        ).astype("float32")

    # Tenure encoding
    tenure_codes = pd.Categorical(
        source["TENURE"],
        categories=config["tenure_order"],
        ordered=True
    ).codes

    out["TENURE_ORDINAL"] = tenure_codes.astype("int8")

        # Region encoding
    region = source["REGION"].fillna("Missing").astype(str)
    region = region.where(
        region.isin(config["region_values"]),
        "Missing"
    )

    for value in config["region_values"]:
        out[f"REGION_{value}"] = (
            region == value
        ).astype("int8")

    out["REGION_Missing"] = (
        region == "Missing"
    ).astype("int8")
    # TOP_PACK frequency encoding
    pack = source["TOP_PACK"].fillna("Other").astype(str)

    pack = pack.where(
        ~pack.isin(set(config["rare_packs"])),
        "Other"
    )

    out["TOP_PACK_FREQ"] = (
        pack.map(config["top_pack_frequency"])
        .fillna(
            config["top_pack_frequency"].get("Other", 0.0)
        )
        .astype("float32")
    )

    # Missingness features
    for column in config["missing_source_columns"]:
        out[f"is_missing_{column}"] = (
            source[column].isna().astype("int8")
        )

    out["MISSING_COUNT"] = (
        source[config["feature_source_columns"]]
        .isna()
        .sum(axis=1)
        .astype("int8")
    )

    # Engineered behavioural ratios
    out["REVENUE_PER_RECHARGE"] = (
        out["REVENUE"]
        / out["FREQUENCE_RECH"].replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).astype("float32")

    out["DATA_SHARE"] = (
        out["DATA_VOLUME"]
        / out["REVENUE"].replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).astype("float32")

    # Log transformations
    for column in config["log_columns"]:
        out[f"LOG1P_{column}"] = (
            np.log1p(
                out[column].clip(lower=0)
            ).astype("float32")
        )

    out = out.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Validate final feature schema
    expected_features = config["feature_columns"]

    missing_features = [
        column
        for column in expected_features
        if column not in out.columns
    ]

    unexpected_features = [
        column
        for column in out.columns
        if column not in expected_features
    ]

    if missing_features or unexpected_features:
        raise ValueError(
            f"Feature schema mismatch. "
            f"Missing={missing_features}; "
            f"unexpected={unexpected_features}"
        )

    return out.reindex(columns=expected_features)


def predict_customer_churn(raw_df, model, config):
    """Return customer IDs and churn probabilities."""

    features = prepare_customer_features(
        raw_df,
        config
    )

    probabilities = model.predict_proba(features)[:, 1]

    result = pd.DataFrame(
        {
            config["id_column"]:
                raw_df[config["id_column"]].to_numpy(),
            "churn_probability":
                probabilities
        }
    )

    return result


def predict_from_bundle(raw_df):
    """Load the saved bundle and predict churn."""

    bundle = load_bundle()

    return predict_customer_churn(
        raw_df=raw_df,
        model=bundle["model"],
        config=bundle["preprocessing_config"]
    )