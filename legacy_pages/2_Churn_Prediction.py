import numpy as np
import pandas as pd
import streamlit as st

def optional_number(label, key):
    value = st.number_input(
        label,
        min_value=0.0,
        value=0.0,
        key=key
    )

    missing = st.checkbox(
        f"{label} is missing",
        key=f"{key}_missing"
    )

    if missing:
        return np.nan

    return value
from app.utils.prediction import load_bundle, predict_customer_churn


st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Individual Customer Churn Prediction")

st.markdown(
    """
    Enter the customer's usage and account information below.
    The application will transform the inputs into the features
    required by the deployed LightGBM churn model.
    """
)

# Load model configuration
@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()
config = bundle["preprocessing_config"]
model = bundle["model"]


with st.form("churn_prediction_form"):

    st.subheader("Customer Details")

    customer_id = st.text_input(
        "Customer ID",
        placeholder="Enter customer identifier"
    )

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox(
            "Region",
            options=config["region_values"]
        )

    with col2:
        tenure = st.selectbox(
            "Tenure",
            options=config["tenure_order"]
        )

    st.subheader("Recharge and Revenue")

    col1, col2, col3 = st.columns(3)

    with col1:
        montant = st.number_input(
            "Recharge Amount (MONTANT)",
            min_value=0.0,
            value=0.0
        )

    with col2:
        frequence_rech = st.number_input(
            "Recharge Frequency (FREQUENCE_RECH)",
            min_value=0.0,
            value=0.0
        )

    with col3:
        revenue = st.number_input(
            "Revenue",
            min_value=0.0,
            value=0.0
        )

    col1, col2 = st.columns(2)

    with col1:
        arpu_segment = st.number_input(
            "ARPU Segment",
            min_value=0.0,
            value=0.0
        )

    with col2:
        frequence = st.number_input(
            "Activity Frequency (FREQUENCE)",
            min_value=0.0,
            value=0.0
        )

    st.subheader("Usage Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:
        data_volume = optional_number(
            "Data Volume",
            "data_volume"
        )    
    with col2:
        on_net = st.number_input(
            "On-Net Usage",
            min_value=0.0,
            value=0.0
        )

    with col3:
        orange = st.number_input(
            "Orange Usage",
            min_value=0.0,
            value=0.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        tigo = st.number_input(
            "Tigo Usage",
            min_value=0.0,
            value=0.0
        )

    with col2:
        zone1 = optional_number(
            "Zone 1 Usage",
            "zone1"
        )

    with col3:
        zone2 = optional_number(
            "Zone 2 Usage",
            "zone2"
        )

    st.subheader("Customer Activity")

    col1, col2 = st.columns(2)

    with col1:
        regularity = st.number_input(
            "Regularity",
            min_value=0.0,
            value=0.0
        )

    with col2:
        freq_top_pack = st.number_input(
            "Top Pack Frequency",
            min_value=0.0,
            value=0.0
        )

    col1, col2 = st.columns(2)

    with col1:
        mrg = st.text_input(
            "MRG",
            value="NO"
        )

    with col2:
        top_pack = st.selectbox(
            "Top Pack",
            options=list(
                config["top_pack_frequency"].keys()
            )
        )

    submitted = st.form_submit_button(
        "Predict Churn Probability",
        type="primary",
        use_container_width=True
    )


if submitted:

    if not customer_id.strip():
        st.error("Please enter a Customer ID.")

    else:

        customer = pd.DataFrame(
            [{
                "user_id": customer_id.strip(),
                "REGION": region,
                "TENURE": tenure,
                "MONTANT": montant,
                "FREQUENCE_RECH": frequence_rech,
                "REVENUE": revenue,
                "ARPU_SEGMENT": arpu_segment,
                "FREQUENCE": frequence,
                "DATA_VOLUME": data_volume,
                "ON_NET": on_net,
                "ORANGE": orange,
                "TIGO": tigo,
                "ZONE1": zone1,
                "ZONE2": zone2,
                "MRG": mrg,
                "REGULARITY": regularity,
                "TOP_PACK": top_pack,
                "FREQ_TOP_PACK": freq_top_pack
            }]
        )

        try:

            result = predict_customer_churn(
                customer,
                model,
                config
            )

            probability = float(
                result.loc[0, "churn_probability"]
            )

            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}"
                )

            with col2:
                st.metric(
                    "Retention Probability",
                    f"{1 - probability:.2%}"
                )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.info(
                "The displayed probability is the model output. "
                "Risk categories have not been assigned because "
                "business risk thresholds have not yet been approved."
            )

        except Exception as exc:

            st.error(
                f"Prediction could not be generated: {exc}"
            )


st.divider()

st.caption(
    "Model: LightGBM | "
    "Validation ROC-AUC: 0.9317 | "
    "Prediction results are intended for decision support."
)