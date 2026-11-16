import io

import pandas as pd
import streamlit as st

from app.utils.prediction import load_bundle, predict_customer_churn


st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Batch Churn Prediction")

st.markdown(
    """
    Upload customer data in CSV format to generate churn probabilities
    for multiple customers using the deployed LightGBM model.
    """
)


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()
model = bundle["model"]
config = bundle["preprocessing_config"]

required_columns = [
    config["id_column"]
] + config["raw_source_columns"]


# ---------------------------------------------------------
# Required columns
# ---------------------------------------------------------

with st.expander("View required CSV columns"):
    st.write(
        "The uploaded CSV must contain the following columns:"
    )

    st.code(
        "\n".join(required_columns)
    )


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

st.divider()
st.subheader("Upload Customer Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        customer_data = pd.read_csv(uploaded_file)

        st.success(
            f"CSV loaded successfully: "
            f"{len(customer_data):,} customer rows."
        )

        st.subheader("Data Preview")

        st.dataframe(
            customer_data.head(10),
            use_container_width=True,
            hide_index=True
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in customer_data.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded CSV is missing the following "
                "required columns:"
            )

            st.write(
                ", ".join(missing_columns)
            )

        else:

            st.info(
                "The CSV contains all columns required by "
                "the deployed model."
            )

            predict_button = st.button(
                "Generate Batch Predictions",
                type="primary",
                use_container_width=True
            )

            if predict_button:

                with st.spinner(
                    "Generating churn probabilities..."
                ):

                    predictions = predict_customer_churn(
                        customer_data,
                        model,
                        config
                    )

                st.success(
                    f"Predictions generated for "
                    f"{len(predictions):,} customers."
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    predictions.head(25),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Prediction Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Customers Scored",
                        f"{len(predictions):,}"
                    )

                with col2:
                    st.metric(
                        "Average Churn Probability",
                        f"{predictions['churn_probability'].mean():.2%}"
                    )

                with col3:
                    st.metric(
                        "Maximum Churn Probability",
                        f"{predictions['churn_probability'].max():.2%}"
                    )

                output = predictions.copy()

                output["churn_probability"] = (
                    output["churn_probability"].round(6)
                )

                csv_buffer = io.StringIO()

                output.to_csv(
                    csv_buffer,
                    index=False
                )

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_buffer.getvalue(),
                    file_name="expresso_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.caption(
                    "No Low, Medium or High risk labels are "
                    "assigned because business risk thresholds "
                    "have not yet been approved."
                )

    except Exception as exc:

        st.error(
            f"The uploaded file could not be processed: {exc}"
        )


st.divider()

st.caption(
    "The batch prediction page uses the same LightGBM model "
    "and preprocessing pipeline as the individual prediction page."
)