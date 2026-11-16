from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Reports",
    page_icon="📑",
    layout="wide"
)

st.title("📑 Reports & Downloads")

st.markdown(
    """
    Access model documentation, customer segmentation summaries
    and supporting deployment information from the Expresso
    Customer Churn project.
    """
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


@st.cache_data
def load_metadata():
    with open(
        ARTIFACT_DIR / "model_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


@st.cache_data
def load_segments():
    return pd.read_csv(
        ARTIFACT_DIR / "segment_profiles.csv"
    )


@st.cache_data
def load_features():
    with open(
        ARTIFACT_DIR / "feature_columns.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


metadata = load_metadata()
segments = load_segments()
features = load_features()


# ---------------------------------------------------------
# Project summary
# ---------------------------------------------------------

st.divider()
st.subheader("Project Summary")

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
        "Training Customers",
        f"{metadata['training_rows']:,}"
    )

with col4:
    st.metric(
        "Model Features",
        metadata["feature_count"]
    )


# ---------------------------------------------------------
# Available reports
# ---------------------------------------------------------

st.divider()
st.subheader("Available Reports")


# Model metadata
with st.container(border=True):

    st.markdown("### Model Metadata")

    st.write(
        "Technical information about the deployed LightGBM "
        "model, validation performance and software versions."
    )

    metadata_json = json.dumps(
        metadata,
        indent=2
    )

    st.download_button(
        "Download Model Metadata",
        data=metadata_json,
        file_name="model_metadata.json",
        mime="application/json",
        use_container_width=True
    )


# Segment profiles
with st.container(border=True):

    st.markdown("### Customer Segment Profiles")

    st.write(
        "Summary of customer volumes, churn rates, revenue "
        "contribution and behavioural characteristics by segment."
    )

    segment_csv = segments.to_csv(
        index=False
    )

    st.download_button(
        "Download Segment Profiles",
        data=segment_csv,
        file_name="segment_profiles.csv",
        mime="text/csv",
        use_container_width=True
    )


# Model features
with st.container(border=True):

    st.markdown("### Model Feature List")

    st.write(
        f"The deployed model uses {len(features)} engineered features."
    )

    feature_text = "\n".join(features)

    st.download_button(
        "Download Feature List",
        data=feature_text,
        file_name="model_features.txt",
        mime="text/plain",
        use_container_width=True
    )


# ---------------------------------------------------------
# Deployment summary
# ---------------------------------------------------------

st.divider()
st.subheader("Deployment Summary")

summary = f"""
EXPRESSO CUSTOMER CHURN PREDICTION

Model:
{metadata['model_name']}

Validation ROC-AUC:
{metadata['validation_roc_auc']:.4f}

Training Rows:
{metadata['training_rows']:,}

Model Features:
{metadata['feature_count']}

Best Iteration:
{metadata['best_iteration']}

Python Version:
{metadata['python_version']}

LightGBM Version:
{metadata['lightgbm_version']}

Risk Thresholds Approved:
{metadata['risk_thresholds_approved']}

Deployment Artifact Created:
{metadata['created_utc']}
"""

st.text(summary)

st.download_button(
    "Download Deployment Summary",
    data=summary,
    file_name="expresso_deployment_summary.txt",
    mime="text/plain",
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction report guidance
# ---------------------------------------------------------

st.divider()
st.subheader("Prediction Reports")

st.info(
    "Individual predictions are displayed on the Churn Prediction "
    "page. Batch Prediction provides downloadable CSV results for "
    "uploaded customer datasets."
)

if not metadata["risk_thresholds_approved"]:

    st.warning(
        "Low, Medium and High churn-risk classifications are "
        "not included because business risk thresholds have "
        "not yet been approved."
    )

st.caption(
    "All downloadable information on this page is generated "
    "from the deployment artifacts packaged with the capstone project."
)