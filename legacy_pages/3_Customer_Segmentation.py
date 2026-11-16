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

st.subheader("Compare Customer Segments")

segment_options = profiles["segment_name"].tolist()

col1, col2 = st.columns(2)

with col1:
    selected_name = st.selectbox(
        "Primary Segment",
        options=segment_options,
        index=0
    )

with col2:
    comparison_name = st.selectbox(
        "Comparison Segment",
        options=segment_options,
        index=1
    )

selected = profiles.loc[
    profiles["segment_name"] == selected_name
].iloc[0]

comparison_segment = profiles.loc[
    profiles["segment_name"] == comparison_name
].iloc[0]

# ---------------------------------------------------------
# Side-by-side segment comparison
# ---------------------------------------------------------

st.divider()
st.subheader("Selected Segment Comparison")

primary_col, comparison_col = st.columns(2)

with primary_col:

    st.markdown(f"### {selected_name}")

    p1, p2 = st.columns(2)

    with p1:
        st.metric(
            "Customers",
            f"{int(selected['customers']):,}"
        )

        st.metric(
            "Churn Rate",
            f"{selected['churn_rate']:.2f}%"
        )

        st.metric(
            "Revenue Share",
            f"{selected['pct_of_revenue']:.1f}%"
        )

    with p2:
        st.metric(
            "Customer Share",
            f"{selected['pct_customers']:.1f}%"
        )

        st.metric(
            "Median Active Days",
            f"{selected['median_active_days']:.0f}"
        )

        st.metric(
            "Data Users",
            f"{selected['data_users']:.1f}%"
        )


with comparison_col:

    st.markdown(f"### {comparison_name}")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Customers",
            f"{int(comparison_segment['customers']):,}"
        )

        st.metric(
            "Churn Rate",
            f"{comparison_segment['churn_rate']:.2f}%"
        )

        st.metric(
            "Revenue Share",
            f"{comparison_segment['pct_of_revenue']:.1f}%"
        )

    with c2:
        st.metric(
            "Customer Share",
            f"{comparison_segment['pct_customers']:.1f}%"
        )

        st.metric(
            "Median Active Days",
            f"{comparison_segment['median_active_days']:.0f}"
        )

        st.metric(
            "Data Users",
            f"{comparison_segment['data_users']:.1f}%"
        )

# ---------------------------------------------------------
# Interactive metric comparison
# ---------------------------------------------------------

st.divider()
st.subheader("Interactive Metric Comparison")

metric_options = {
    "Churn Rate": ("churn_rate", "%"),
    "Revenue Share": ("pct_of_revenue", "%"),
    "Customer Share": ("pct_customers", "%"),
    "Share of Churners": ("pct_of_churners", "%"),
    "Median Revenue": ("median_revenue", ""),
    "Median Active Days": ("median_active_days", ""),
    "Data Users": ("data_users", "%")
}

selected_metric = st.selectbox(
    "Choose a metric to compare",
    options=list(metric_options.keys())
)

metric_column, metric_suffix = metric_options[selected_metric]

comparison_chart = pd.DataFrame(
    {
        "Segment": [
            selected_name,
            comparison_name
        ],
        selected_metric: [
            selected[metric_column],
            comparison_segment[metric_column]
        ]
    }
).set_index("Segment")

st.bar_chart(
    comparison_chart
)


# Display exact values below the chart
left_value, right_value = st.columns(2)

with left_value:
    if metric_suffix == "%":
        st.metric(
            selected_name,
            f"{selected[metric_column]:.2f}%"
        )
    else:
        st.metric(
            selected_name,
            f"{selected[metric_column]:,.0f}"
        )

with right_value:
    if metric_suffix == "%":
        st.metric(
            comparison_name,
            f"{comparison_segment[metric_column]:.2f}%"
        )
    else:
        st.metric(
            comparison_name,
            f"{comparison_segment[metric_column]:,.0f}"
        )
# ---------------------------------------------------------
# Portfolio benchmarks
# ---------------------------------------------------------

st.divider()
st.subheader("Portfolio Benchmark")

total_customers = profiles["customers"].sum()

portfolio_churn_rate = (
    (
        profiles["customers"]
        * profiles["churn_rate"]
    ).sum()
    / total_customers
)

portfolio_active_days = (
    (
        profiles["customers"]
        * profiles["median_active_days"]
    ).sum()
    / total_customers
)

portfolio_data_users = (
    (
        profiles["customers"]
        * profiles["data_users"]
    ).sum()
    / total_customers
)


primary_benchmark, comparison_benchmark = st.columns(2)


with primary_benchmark:

    st.markdown(f"### {selected_name}")

    st.metric(
        "Churn Rate vs Portfolio",
        f"{selected['churn_rate']:.2f}%",
        delta=(
            f"{selected['churn_rate'] - portfolio_churn_rate:+.2f} pp"
        ),
        delta_color="inverse"
    )

    st.metric(
        "Median Active Days vs Portfolio",
        f"{selected['median_active_days']:.0f}",
        delta=(
            f"{selected['median_active_days'] - portfolio_active_days:+.1f}"
        )
    )

    st.metric(
        "Data Users vs Portfolio",
        f"{selected['data_users']:.1f}%",
        delta=(
            f"{selected['data_users'] - portfolio_data_users:+.1f} pp"
        )
    )


with comparison_benchmark:

    st.markdown(f"### {comparison_name}")

    st.metric(
        "Churn Rate vs Portfolio",
        f"{comparison_segment['churn_rate']:.2f}%",
        delta=(
            f"{comparison_segment['churn_rate'] - portfolio_churn_rate:+.2f} pp"
        ),
        delta_color="inverse"
    )

    st.metric(
        "Median Active Days vs Portfolio",
        f"{comparison_segment['median_active_days']:.0f}",
        delta=(
            f"{comparison_segment['median_active_days'] - portfolio_active_days:+.1f}"
        )
    )

    st.metric(
        "Data Users vs Portfolio",
        f"{comparison_segment['data_users']:.1f}%",
        delta=(
            f"{comparison_segment['data_users'] - portfolio_data_users:+.1f} pp"
        )
    )


st.caption(
    f"Portfolio benchmarks: churn rate {portfolio_churn_rate:.2f}%, "
    f"weighted median-active-days benchmark {portfolio_active_days:.1f}, "
    f"and weighted data-user benchmark {portfolio_data_users:.1f}%."
)
# ---------------------------------------------------------
# Management interpretation
# ---------------------------------------------------------

st.divider()
st.subheader("Management Interpretation")


def management_interpretation(segment, segment_name):

    churn_gap = (
        segment["churn_rate"]
        - portfolio_churn_rate
    )

    active_gap = (
        segment["median_active_days"]
        - portfolio_active_days
    )

    data_gap = (
        segment["data_users"]
        - portfolio_data_users
    )

    revenue_share = segment["pct_of_revenue"]
    customer_share = segment["pct_customers"]

    insights = []

    # Churn interpretation
    if churn_gap > 5:
        insights.append(
            f"Churn is {churn_gap:.2f} percentage points "
            f"above the portfolio benchmark, indicating a "
            f"priority segment for retention or reactivation."
        )
    elif churn_gap < -5:
        insights.append(
            f"Churn is {abs(churn_gap):.2f} percentage points "
            f"below the portfolio benchmark, indicating relatively "
            f"strong customer retention."
        )
    else:
        insights.append(
            f"Churn is broadly aligned with the portfolio benchmark "
            f"({churn_gap:+.2f} percentage points)."
        )

    # Engagement interpretation
    if active_gap < -5:
        insights.append(
            f"Customer activity is below the portfolio benchmark "
            f"by {abs(active_gap):.1f} active days, indicating "
            f"weaker engagement."
        )
    elif active_gap > 5:
        insights.append(
            f"Customer activity is above the portfolio benchmark "
            f"by {active_gap:.1f} active days, indicating "
            f"stronger engagement."
        )

    # Data adoption interpretation
    if data_gap < -10:
        insights.append(
            f"Data adoption is {abs(data_gap):.1f} percentage "
            f"points below the portfolio benchmark, suggesting "
            f"potential for targeted data-bundle adoption campaigns."
        )
    elif data_gap > 10:
        insights.append(
            f"Data adoption is {data_gap:.1f} percentage points "
            f"above the portfolio benchmark, indicating a "
            f"strong digitally engaged customer group."
        )

    # Revenue concentration interpretation
    revenue_gap = revenue_share - customer_share

    if revenue_gap > 10:
        insights.append(
            f"The segment represents {customer_share:.1f}% of "
            f"customers but contributes {revenue_share:.1f}% of "
            f"revenue, making value protection a management priority."
        )
    elif revenue_gap < -10:
        insights.append(
            f"The segment represents {customer_share:.1f}% of "
            f"customers but contributes only {revenue_share:.1f}% "
            f"of revenue, indicating relatively low commercial value."
        )

    return insights


primary_insights = management_interpretation(
    selected,
    selected_name
)

comparison_insights = management_interpretation(
    comparison_segment,
    comparison_name
)


left_insight, right_insight = st.columns(2)


with left_insight:

    st.markdown(f"### {selected_name}")

    for insight in primary_insights:

        if (
            selected["churn_rate"]
            > portfolio_churn_rate + 5
        ):
            st.warning(insight)
        else:
            st.info(insight)


with right_insight:

    st.markdown(f"### {comparison_name}")

    for insight in comparison_insights:

        if (
            comparison_segment["churn_rate"]
            > portfolio_churn_rate + 5
        ):
            st.warning(insight)
        else:
            st.info(insight)


st.caption(
    "Management interpretations are generated from observed segment "
    "characteristics relative to portfolio benchmarks. They support "
    "prioritisation but do not replace business judgement or approved "
    "customer-treatment policies."
)
# ---------------------------------------------------------
# Recommended management actions
# ---------------------------------------------------------

st.divider()
st.subheader("Recommended Management Actions")

segment_recommendations = {
    "Silent / Inactive": [
        "Prioritize targeted reactivation campaigns for customers with previous value potential.",
        "Investigate the main causes of inactivity before applying retention incentives.",
        "Use controlled recharge or usage incentives to test customer re-engagement.",
        "Differentiate potentially recoverable customers from persistently inactive, zero-value accounts."
    ],

    "High-Value Active": [
        "Prioritize value protection because this segment contributes a disproportionate share of portfolio revenue.",
        "Use proactive service-recovery interventions when activity or recharge behaviour starts declining.",
        "Consider loyalty benefits and personalized offers to reinforce retention.",
        "Monitor changes in regularity, recharge frequency and usage as potential early-warning indicators."
    ],

    "Data-Engaged": [
        "Strengthen retention through targeted data-bundle and digital-service offers.",
        "Use cross-selling opportunities that complement existing high data adoption.",
        "Encourage higher recharge frequency and longer-duration data packages.",
        "Monitor declining data usage as a potential early sign of disengagement."
    ],

    "Light Active": [
        "Increase customer engagement through affordable targeted bundles and recharge incentives.",
        "Promote migration toward higher-value products based on observed usage behaviour.",
        "Use personalized campaigns to increase activity frequency and active days.",
        "Monitor customers whose activity declines toward the Low-Engagement segment."
    ],

    "Low-Engagement": [
        "Prioritize early retention intervention before customers become fully inactive.",
        "Use targeted reactivation or low-cost engagement offers.",
        "Investigate barriers to recharge, data adoption and regular usage.",
        "Monitor migration toward the Silent / Inactive segment as an early-warning signal."
    ]
}


def show_recommendations(segment_name):

    recommendations = segment_recommendations.get(
        segment_name,
        []
    )

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):
        st.write(
            f"{number}. {recommendation}"
        )


recommendation_left, recommendation_right = st.columns(2)


with recommendation_left:

    st.markdown(f"### {selected_name}")

    show_recommendations(
        selected_name
    )


with recommendation_right:

    st.markdown(f"### {comparison_name}")

    show_recommendations(
        comparison_name
    )


st.caption(
    "Recommended actions are management strategies derived from "
    "the observed behavioural characteristics of each segment. "
    "They are not outputs generated by the churn model and should "
    "be validated against campaign economics, customer-treatment "
    "policies and operational capacity before implementation."
)