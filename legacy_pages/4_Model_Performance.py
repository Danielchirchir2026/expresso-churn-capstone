from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Performance")

st.markdown(
    """
    Review the performance, configuration and most influential
    features of the deployed Expresso churn prediction model.
    """
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


@st.cache_resource
def load_model_bundle():
    return joblib.load(
        ARTIFACT_DIR / "expresso_churn_bundle.joblib"
    )


@st.cache_data
def load_metadata():
    with open(
        ARTIFACT_DIR / "model_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


bundle = load_model_bundle()
metadata = load_metadata()
model = bundle["model"]


# ---------------------------------------------------------
# Performance overview
# ---------------------------------------------------------

st.divider()
st.subheader("Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        metadata["model_name"]
    )

with col2:
    st.metric(
        "Validation ROC-AUC",
        f"{metadata['validation_roc_auc']:.4f}"
    )

with col3:
    st.metric(
        "Model Features",
        metadata["feature_count"]
    )

with col4:
    st.metric(
        "Best Iteration",
        metadata["best_iteration"]
    )


# ---------------------------------------------------------
# Training information
# ---------------------------------------------------------

st.divider()
st.subheader("Training Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Training Rows",
        f"{metadata['training_rows']:,}"
    )

with col2:
    st.metric(
        "Validation Seed",
        metadata["validation_split_seed"]
    )

with col3:
    st.metric(
        "LightGBM Version",
        metadata["lightgbm_version"]
    )


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

st.divider()
st.subheader("Feature Importance")

importance = pd.DataFrame(
    {
        "Feature": model.feature_name_,
        "Importance": model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=False
)

top_n = st.slider(
    "Number of features to display",
    min_value=5,
    max_value=30,
    value=15,
    step=5
)

top_features = importance.head(top_n)

st.bar_chart(
    top_features.set_index("Feature")
)

with st.expander("View feature importance table"):
    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# Model environment
# ---------------------------------------------------------

st.divider()
st.subheader("Deployment Environment")

environment = pd.DataFrame(
    {
        "Component": [
            "Python",
            "NumPy",
            "Pandas",
            "Scikit-learn",
            "LightGBM"
        ],
        "Version": [
            metadata["python_version"],
            metadata["numpy_version"],
            metadata["pandas_version"],
            metadata["scikit_learn_version"],
            metadata["lightgbm_version"]
        ]
    }
)

st.dataframe(
    environment,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Validation notes
# ---------------------------------------------------------

st.divider()
st.subheader("Model Validation Notes")

st.success(
    "The deployed LightGBM model achieved a validation "
    f"ROC-AUC of {metadata['validation_roc_auc']:.4f}."
)

if not metadata["risk_thresholds_approved"]:
    st.warning(
        "Business thresholds for classifying customers into "
        "Low, Medium or High churn-risk categories have not "
        "been formally approved. The application therefore "
        "reports churn probabilities rather than assigning "
        "risk labels."
    )

st.info(
    "ROC curves and confusion-matrix statistics will only be "
    "displayed after the corresponding validation predictions "
    "and labels are added to the deployment artifacts."
)

st.caption(
    "Model performance information is sourced from the saved "
    "deployment metadata and LightGBM model."
)