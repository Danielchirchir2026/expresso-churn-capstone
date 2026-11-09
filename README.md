# 📊 Expresso Customer Churn Prediction & Retention Analytics

> **A machine-learning and customer analytics capstone project for predicting customer churn, understanding customer behaviour, and supporting targeted retention strategies.**

---

## 🚀 Live Application

### Live App: churn-prediction-capstone.streamlit.app

The solution is deployed on **Streamlit Community Cloud** and provides interactive churn analytics, customer segmentation, prediction, batch scoring, and management reporting.

---

## 👥 Project Team

### **Spartans**

1. **Daniel Chirchir**
2. **Daniel Tabut**
3. **Livingstone Talel**
4. **Sandra Bosibori**
5. **Patrick Palex**
6. **Immanuel Ochieng**

---

## 🎯 Project Objective

Customer churn represents a significant challenge for telecommunications businesses because customer attrition can affect:

- Revenue
- Customer lifetime value
- Customer acquisition costs
- Product adoption
- Long-term business growth

The project develops a data-driven framework for:

- 🎯 Predicting individual **customer churn probability**
- 🔍 Identifying the most important **churn drivers**
- 👥 Segmenting customers according to **behaviour and value**
- 🔄 Identifying customer groups requiring **retention or reactivation**
- 📊 Supporting management decisions through **interactive analytics**
- 📂 Generating **individual and batch churn predictions**
- 💡 Providing segment-level **management interpretations and recommended actions**

---

## 🤖 Machine Learning Model

The production churn model is a **LightGBM classifier**.

### Model Performance

| Metric | Result |
|---|---:|
| **Validation ROC-AUC** | **0.9317** |
| Training Rows | 2,154,048 |
| Engineered Model Features | 69 |
| Best Iteration | 345 |
| Python Version | 3.12 |
| LightGBM Version | 4.6.0 |

> **Important:** The application reports churn probabilities rather than automatically assigning customers to Low, Medium, or High churn-risk categories because formal business risk thresholds have not yet been approved.

---

## 🧭 Application Modules

### 1. 📈 Executive Dashboard

Provides management with a portfolio-level view of:

- **Total customers**
- **Overall churn rate**
- **Estimated churned customers**
- Customer distribution by segment
- Churn rate by segment
- Revenue contribution by segment
- Share of churners by segment
- Major churn drivers
- Executive management insights

### 2. 🎯 Churn Prediction

Predicts the churn probability of an **individual customer** using:

- Customer tenure and region
- Recharge amount and frequency
- Revenue and ARPU
- Data usage
- Voice usage
- Customer activity and regularity
- Preferred package behaviour

The raw customer data is transformed into the **69 engineered features** required by the LightGBM model.

### 3. 👥 Customer Segmentation

Interactive behavioural segmentation containing:

1. **Silent / Inactive**
2. **High-Value Active**
3. **Data-Engaged**
4. **Light Active**
5. **Low-Engagement**

#### Interactive functionality

- Two independently selectable customer segments
- Side-by-side segment KPIs
- Interactive metric comparison
- Portfolio benchmarking
- Dynamic management interpretation
- Segment-specific recommended actions

### 4. 🤖 Model Performance

Provides transparency into:

- Validation ROC-AUC
- Training observations
- Feature count
- Best iteration
- Feature importance
- Deployment environment

### 5. 📂 Batch Prediction

1. Upload customer data in CSV format.
2. Validate required model columns.
3. Preview customer data.
4. Generate churn probabilities.
5. Review summary statistics.
6. Download prediction results.

### 6. 📑 Reports

Provides downloadable:

- Model metadata
- Customer segment profiles
- Model feature list
- Deployment summary

---

## 📊 Portfolio Insights

| Portfolio Measure | Result |
|---|---:|
| **Customers Analysed** | **2,154,048** |
| **Overall Churn Rate** | **18.75%** |
| **Estimated Churned Customers** | **403,986** |
| **Validation ROC-AUC** | **0.9317** |

### Key Observed Patterns

- **Silent / Inactive** customers show the highest churn exposure.
- **High-Value Active** customers contribute approximately **63% of portfolio revenue** while recording very low observed churn.
- Customer regularity, usage behaviour, recharge patterns, and revenue characteristics are important churn predictors.
- Customer retention strategies should therefore differ materially across behavioural segments.

---

## 🧩 Segment Management Strategies

| Customer Segment | Primary Management Focus |
|---|---|
| **Silent / Inactive** | Reactivation campaigns and investigation of inactivity drivers |
| **High-Value Active** | Revenue protection, loyalty and proactive retention |
| **Data-Engaged** | Data-bundle retention, digital engagement and cross-selling |
| **Light Active** | Increase engagement, recharge frequency and product migration |
| **Low-Engagement** | Early retention intervention before complete inactivity |

### Management Decision Support

The Customer Segmentation module compares each selected segment against portfolio benchmarks covering:

- Churn rate
- Customer activity
- Data adoption
- Revenue contribution
- Customer share

Segments performing materially worse than portfolio benchmarks can be prioritized for **retention or reactivation**, while high-value segments with strong retention can be prioritized for **value protection**.

---

## 🔍 Leading Churn Drivers

The deployed LightGBM model identifies the following among the leading churn drivers:

1. **REGULARITY**
2. **ON_NET**
3. **DATA_VOLUME**
4. **REVENUE_PER_RECHARGE**
5. **ORANGE**
6. **DATA_SHARE**
7. **MONTANT**
8. **FREQUENCE**
9. **TIGO**
10. **REVENUE**

---

## 🛠️ Technology Stack

### Machine Learning & Analytics

- **Python 3.12**
- **LightGBM 4.6.0**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Joblib**

### Application

- **Streamlit**

### Version Control & Collaboration

- **Git**
- **GitHub**

### Deployment

- **Streamlit Community Cloud**

---

