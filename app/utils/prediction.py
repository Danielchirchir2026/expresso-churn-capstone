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
