from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation")

st.markdown(
    """
    Explore the customer groups identified from usage, revenue,
    recharge and engagement behaviour in the Expresso customer base.
    """
)

# Locate project artifacts
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_DIR / "artifacts" / "segment_profiles.csv"


@st.cache_data
def load_segment_profiles():
    return pd.read_csv(PROFILE_PATH)


profiles = load_segment_profiles()


# Descriptive business labels derived from the saved segment profiles
segment_names = {
    0: "Silent / Inactive",
    1: "High-Value Active",
    2: "Data-Engaged",
    3: "Light Active",
    4: "Low-Engagement"
}

profiles["segment_name"] = profiles["segment"].map(segment_names)


# ---------------------------------------------------------
# Overall KPIs
# ---------------------------------------------------------

st.divider()

total_customers = int(profiles["customers"].sum())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Customers Analysed",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Customer Segments",
        len(profiles)
    )

with col3:
    largest = profiles.loc[
        profiles["customers"].idxmax()
    ]

    st.metric(
        "Largest Segment",
        largest["segment_name"]
    )

with col4:
    highest_churn = profiles.loc[
        profiles["churn_rate"].idxmax()
    ]

    st.metric(
        "Highest Churn Segment",
        highest_churn["segment_name"]
    )


# ---------------------------------------------------------
# Segment selector
# ---------------------------------------------------------

st.divider()

st.subheader("Explore a Segment")

selected_name = st.selectbox(
    "Select customer segment",
    options=profiles["segment_name"].tolist()
)

selected = profiles.loc[
    profiles["segment_name"] == selected_name
].iloc[0]


# -------