from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Executive Dashboard")

st.markdown(
    """
    Executive view of customer churn exposure, customer segments,
    revenue concentration and the main drivers of the deployed
    churn prediction model.
    """
)


# ---------------------------------------------------------
# Load deployment artifacts
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


@st.cache_data
def load_segment_profiles():
    return pd.read_csv(
        ARTIFACT_DIR / "segment_profiles.csv"
    )


@st.cache_data
def load_metadata():
    with open(
        ARTIFACT_DIR / "model_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


@st.cache_resource
def load_model():
    bundle = joblib.load(
        ARTIFACT_DIR / "expresso_churn_bundle.joblib"
    )
    return bundle["model"]


profiles = load_segment_profiles()
metadata = load_metadata()
model = load_model()


segment_names = {
    0: "Silent / Inactive",
    1: "High-Value Active",
    2: "Data-Engaged",
    3: "Light Active",
    4: "Low-Engagement"
}

profiles["segment_name"] = (
    profiles["segment"].map(segment_names)
)


# ---------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------

total_customers = int(profiles["customers"].sum())

estimated_churners = (
    profiles["customers"]
    * profiles["churn_rate"]
    / 100
).sum()

portfolio_churn_rate = (
    estimated_churners
    / total_customers
    * 100
)

highest_churn_segment = profiles.loc[
    profiles["churn_rate"].idxmax()
]

highest_revenue_segment = profiles.loc[
    profiles["pct_of_revenue"].idxmax()
]


st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Overall Churn Rate",
        f"{portfolio_churn_rate:.2f}%"
    )

with col3:
    st.metric(
        "Estimated Churned Customers",
        f"{estimated_churners:,.0f}"
    )

with col4:
    st.metric(
        "Model ROC-AUC",
        f"{metadata['validation_roc_auc']:.4f}"
    )


# ---------------------------------------------------------
# Customer and churn distribution
# ---------------------------------------------------------

st.divider()

left, right = st.columns(2)

with left:

    st.subheader("Customer Distribution")

    customer_distribution = (
        profiles[
            ["segment_name", "customers"]
        ]
        .set_index("segment_name")
    )

    st.bar_chart(
        customer_distribution
    )


with right:

    st.subheader("Churn Rate by Segment")

    churn_distribution = (
        profiles[
            ["segment_name", "churn_rate"]
        ]
        .set_index("segment_name")
    )

    st.bar_chart(
        churn_distribution
    )


# ---------------------------------------------------------
# Revenue and churn concentration
# ---------------------------------------------------------

st.divider()

left, right = st.columns(2)

with left:

    st.subheader("Revenue Contribution by Segment")

    revenue_distribution = (
        profiles[
            ["segment_name", "pct_of_revenue"]
        ]
        .set_index("segment_name")
    )

    st.bar_chart(
        revenue_distribution
    )


with right:

    st.subheader("Share of Churners by Segment")

    churner_distribution = (
        profiles[
            ["segment_name", "pct_of_churners"]
        ]
        .set_index("segment_name")
    )

    st.bar_chart(
        churner_distribution
    )


# ---------------------------------------------------------
# Key churn drivers
# ---------------------------------------------------------

st.divider()

st.subheader("Key Churn Drivers")

importance = pd.DataFrame(
    {
        "Feature": model.feature_name_,
        "Importance": model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=False
)

top_drivers = importance.head(10)

st.bar_chart(
    top_drivers.set_index("Feature")
)


# ---------------------------------------------------------
# Executive insights
# ---------------------------------------------------------

st.divider()

st.subheader("Executive Insights")

col1, col2 = st.columns(2)

with col1:

    st.warning(
        f"Highest churn exposure: "
        f"{highest_churn_segment['segment_name']} "
        f"has an observed churn rate of "
        f"{highest_churn_segment['churn_rate']:.2f}%."
    )

with col2:

    st.success(
        f"Revenue concentration: "
        f"{highest_revenue_segment['segment_name']} "
        f"contributes approximately "
        f"{highest_revenue_segment['pct_of_revenue']:.1f}% "
        f"of observed portfolio revenue."
    )


silent = profiles.loc[
    profiles["segment"] == 0
].iloc[0]

high_value = profiles.loc[
    profiles["segment"] == 1
].iloc[0]


st.info(
    f"The Silent / Inactive segment contains "
    f"{int(silent['customers']):,} customers "
    f"({silent['pct_customers']:.1f}% of the portfolio) "
    f"and accounts for {silent['pct_of_churners']:.1f}% "
    f"of observed churners."
)

st.info(
    f"The High-Value Active segment contributes "
    f"{high_value['pct_of_revenue']:.1f}% of portfolio revenue "
    f"while recording an observed churn rate of only "
    f"{high_value['churn_rate']:.2f}%."
)


# ---------------------------------------------------------
# Management table
# ---------------------------------------------------------

st.divider()

st.subheader("Segment Management Summary")

summary = profiles[
    [
        "segment_name",
        "customers",
        "pct_customers",
        "churn_rate",
        "pct_of_churners",
        "pct_of_revenue",
        "median_revenue",
        "median_active_days"
    ]
].copy()

summary.columns = [
    "Segment",
    "Customers",
    "% Customers",
    "Churn Rate %",
    "% of Churners",
    "% of Revenue",
    "Median Revenue",
    "Median Active Days"
]

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Dashboard metrics are derived from the customer segment "
    "profiles and deployment metadata packaged with the capstone. "
    "The overall churn rate is calculated as the customer-count "
    "weighted average of observed segment churn rates."
)