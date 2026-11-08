import streamlit as st

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Executive Dashboard")

st.markdown(
    """
    Monitor the overall customer churn position and identify
    the key factors contributing to churn risk.
    """
)

st.divider()

# KPI section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Customers",
        value="--"
    )

with col2:
    st.metric(
        label="Churn Rate",
        value="--"
    )

with col3:
    st.metric(
        label="Customers at Risk",
        value="--"
    )

with col4:
    st.metric(
        label="Average Churn Probability",
        value="--"
    )

st.divider()

# Dashboard visualisation section
left, right = st.columns(2)

with left:
    st.subheader("Customer Distribution")
    st.info("Customer distribution visualization will appear here.")

with right:
    st.subheader("Churn Distribution")
    st.info("Churn distribution visualization will appear here.")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Churn Risk by Customer Segment")
    st.info("Customer segment analysis will appear here.")

with right:
    st.subheader("Key Churn Drivers")
    st.info("Key churn driver analysis will appear here.")

st.divider()

st.subheader("Executive Insights")

st.info(
    "Key management insights will be generated from the "
    "customer data and model predictions."
)