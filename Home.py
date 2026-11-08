import streamlit as st

st.set_page_config(
    page_title="Expresso Customer Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Expresso Customer Churn Analytics")

st.markdown(
    """
    Welcome to the **Expresso Customer Churn Analytics & Prediction System**.

    This application transforms the results of the Expresso churn
    capstone project into an interactive decision-support tool.

    ### Available Modules

    **📈 Executive Dashboard**  
    View overall churn risk, customer distribution, and key insights.

    **🎯 Churn Prediction**  
    Predict the churn probability of an individual customer.

    **👥 Customer Segmentation**  
    Explore customer groups and their churn characteristics.

    **🤖 Model Performance**  
    Review model performance, including AUC, ROC curve,
    confusion matrix, and feature importance.

    **📂 Batch Prediction**  
    Upload customer data in CSV format and generate churn predictions.

    **📑 Reports**  
    Download prediction results and summary reports.
    """
)

st.divider()

st.info(
    "Select a module from the sidebar to begin."
)

st.caption("Expresso Customer Churn Capstone Project")