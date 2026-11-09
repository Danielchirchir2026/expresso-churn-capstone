Expresso Customer Churn Prediction and Retention Analytics

A machine-learning and customer analytics capstone project developed to identify customers at risk of churn, understand customer behaviour, and support targeted retention strategies.



Live Application

Streamlit App:

https://churn-prediction-capstone.streamlit.app



Project Team

Spartans

Daniel Chirchir

Daniel Tabut

Livingstone Talel

Sandra Bosibori

Patrick Palex

Immanuel Ochieng

Project Objective

Customer churn represents a significant challenge for telecommunications businesses because losing existing customers affects revenue, customer lifetime value, and future growth.



This project develops a data-driven framework for:



Predicting customer churn probability.

Identifying the major factors associated with churn.

Segmenting customers according to behavioural characteristics.

Identifying customer groups requiring retention or reactivation.

Supporting management decisions through interactive analytics.

Generating individual and batch churn predictions.

Machine Learning Model

The deployed churn model is a LightGBM classifier.



Metric	Result

Validation ROC-AUC	0.9317

Training Rows	2,154,048

Engineered Model Features	69

Best Iteration	345

The application reports churn probabilities rather than automatically assigning Low, Medium, or High risk categories because business risk thresholds have not been formally approved.



Application Modules

1\. Executive Dashboard

Provides management with an overall view of the customer portfolio, including:



Total customers

Overall churn rate

Estimated churned customers

Model ROC-AUC

Customer distribution by segment

Churn rate by segment

Revenue contribution by segment

Share of churners by segment

Major churn drivers

Executive management insights

2\. Churn Prediction

Predicts churn probability for an individual customer using customer account, recharge, revenue, usage, and activity information.



The application transforms the raw customer inputs into the engineered features required by the deployed LightGBM model.



3\. Customer Segmentation

Provides an interactive analysis of behavioural customer groups.



The deployed segmentation analysis contains:



Silent / Inactive

High-Value Active

Data-Engaged

Light Active

Low-Engagement

The page includes:



Side-by-side segment comparison

Interactive metric comparison

Portfolio benchmarking

Dynamic management interpretation

Segment-specific management recommendations

4\. Model Performance

Provides information about the deployed machine-learning model, including:



Validation ROC-AUC

Training observations

Model feature count

Best LightGBM iteration

Feature importance

Deployment environment information

5\. Batch Prediction

Allows users to upload customer data in CSV format and generate churn probabilities for multiple customers simultaneously.



Prediction results can be downloaded as a CSV file for further analysis.



6\. Reports

Provides downloadable project and model documentation, including:



Model metadata

Customer segment profiles

Model feature list

Deployment summary

Customer Segments

Segment	Management Focus

Silent / Inactive	Reactivation and investigation of inactivity drivers

High-Value Active	Revenue protection, loyalty and proactive retention

Data-Engaged	Data-bundle retention, digital engagement and cross-selling

Light Active	Increase usage, recharge frequency and product migration

Low-Engagement	Early retention intervention before complete inactivity

Technology Stack

Python 3.12

Streamlit

LightGBM

Scikit-learn

Pandas

NumPy

Joblib

Git

GitHub

Streamlit Community Cloud

Repository Structure

EXPRESSO\_CHURN\_CAPSTONE

|

|-- Home.py

|-- requirements.txt

|-- README.md

|

|-- app/

|   `-- utils/

|       `-- prediction.py

|

|-- artifacts/

|   |-- expresso\_churn\_bundle.joblib

|   |-- feature\_columns.json

|   |-- model\_metadata.json

|   |-- preprocessing\_config.joblib

|   `-- segment\_profiles.csv

|

`-- pages/

&#x20;   |-- 1\_Executive\_Dashboard.py

&#x20;   |-- 2\_Churn\_Prediction.py

&#x20;   |-- 3\_Customer\_Segmentation.py

&#x20;   |-- 4\_Model\_Performance.py

&#x20;   |-- 5\_Batch\_Prediction.py

&#x20;   `-- 6\_Reports.py



Deployment



The application is deployed through Streamlit Community Cloud from the main branch of the GitHub repository.



Application entry point:





Home.py



Python deployment version:





3.12

Important Interpretation Note



Churn probabilities are decision-support outputs and should be interpreted together with relevant customer and business information.



Segment-level management recommendations are derived from observed behavioural profiles and portfolio benchmarks. They are management recommendations rather than direct outputs of the churn model and should be validated against campaign economics, customer-treatment policies, and operational capacity before implementation.



Group



Spartans



Expresso Customer Churn Prediction and Retention Analytics Capstone Project, 2026.





\### Save the README



Save and close Notepad.



Then verify that Git sees both the committed Home change and the new README appropriately:



```cmd

git status --short

