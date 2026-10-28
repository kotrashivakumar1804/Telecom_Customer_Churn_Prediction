# 📡 Telecom Customer Churn Prediction System

An interactive Machine Learning web application built with **Streamlit**, **Python**, and **XGBoost** to predict whether a telecom customer is likely to churn (cancel their service) or remain loyal.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How to Use the App](#-how-to-use-the-app)
- [Business Impact](#-business-impact)

---

## 🌟 Overview
In the telecommunications industry, acquiring new customers costs significantly more than retaining existing ones. Customer churn directly impacts revenue and market share. 

This project provides an automated predictive solution that analyzes **17 customer attributes** (demographic profile, subscribed services, contract details, and charges) to calculate the likelihood of churn in real-time, accompanied by confidence probability scores.

---

## ✨ Key Features
* **Real-Time Prediction**: Instantly predicts whether a customer will stay or churn upon form submission.
* **Confidence Metric**: Calculates prediction probability using `predict_proba()` to guide business priority.
* **Modern Cyber-Themed UI**: Sleek dark mode, custom gold and neon green visual accents, custom SVG network tower, and glassmorphism cards.
* **Intelligent Data Imputation**: Handles missing values in `TotalCharges` using hierarchical group-based means.
* **Form Reset**: Quick reset button to clear inputs and prepare for new customer evaluations.

---

## 🧠 Machine Learning Pipeline

1. **Dataset**: [Telco Customer Churn dataset](Telco-Customer-Churn.csv) (7,043 customer entries).
2. **Data Cleaning & Missing Value Handling**:
   - Converted `TotalCharges` from object/string to numeric.
   - Imputed missing values through group-level averages based on customer service profiles.
3. **Feature Encoding & Scaling**:
   - Binary encoding for Partner, Dependents, PhoneService, PaperlessBilling.
   - Categorical label mapping for InternetService, Streaming services, Contract, and PaymentMethod.
   - Normalized continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`).
4. **Model Architecture**:
   - **Algorithm**: `XGBClassifier` (Extreme Gradient Boosting)
   - **Hyperparameters**: `n_estimators=100`, `max_depth=3`
   - **Split**: 70% Training / 30% Testing (`random_state=42`)

---

## 📊 Model Evaluation & Results

Evaluated on an unseen test set of **2,113 customer records**:

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Testing Accuracy** | `0.7989` | **~80%** |
| **Training Accuracy** | `0.8357` | **~83.6%** |
| **ROC-AUC Score** | `0.8492` | **84.92%** |
| **Precision (Churners)** | `0.6602` | **66.02%** |
| **Recall (Churners)** | `0.5348` | **53.48%** |
| **F1-Score (Churners)** | `0.5910` | **59.10%** |

### Confusion Matrix
```text
               Predicted: Stay    Predicted: Churn
Actual: Stay        1,381               158
Actual: Churn         267               307
```

---

## 🛠️ Tech Stack
* **Language**: Python 3.10+
* **ML Libraries**: XGBoost, Scikit-Learn
* **Data Processing**: Pandas, NumPy
* **Visualization & Frontend**: Streamlit, Matplotlib, Seaborn
* **Styling**: Custom CSS3, Inline SVGs

---

## 📁 Project Structure

```text
Telecom Customer Churn prediction/
│
├── ML_Project.py              # Main Streamlit web application & ML workflow
├── Telco-Customer-Churn.csv   # Telecom customer dataset
├── style.css                  # Custom styling and layout design
├── README.md                  # Comprehensive project documentation
└── .gitignore                 # Files excluded from version control
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install streamlit pandas numpy xgboost scikit-learn seaborn matplotlib
```

### 3. Run the Web Application
```bash
streamlit run ML_Project.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🖥️ How to Use the App
1. **Section 1 - Customer**: Select Partner status, Dependents, and input Tenure in months.
2. **Section 2 - Services**: Specify subscribed services (Internet, Phone, Security, Tech Support, Streaming, etc.).
3. **Section 3 - Billing**: Select Contract duration (Month-to-month, 1-year, 2-year), Payment Method, and enter Monthly and Total Charges.
4. Click **🚀 Predict** to view the churn classification result and confidence percentage.
5. Click **🔄 Reset** to clear inputs.

---

## 💼 Business Impact
* **Targeted Retention**: Empowers customer success teams to identify at-risk customers early and offer tailored incentives (contract extensions, discounts, service bundles).
* **Cost Efficiency**: Decreases customer acquisition expenses by reducing churn rates.
* **Actionable Insights**: Highlights that contract duration (Month-to-month vs. Long-term) and Tech Support availability are key drivers of customer retention.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
