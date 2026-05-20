# 💳 Credit Card Fraud Detection using Ensemble Learning

An end-to-end Machine Learning system designed to identify fraudulent credit card transactions using **ensemble learning techniques** and provide **real-time fraud analysis** through an interactive Streamlit application.

The project covers the complete ML workflow including:

✔ Data preprocessing  
✔ Exploratory Data Analysis (EDA)  
✔ Feature Engineering  
✔ Model Benchmarking  
✔ Model Evaluation  
✔ Hyperparameter Tuning  
✔ Model Deployment using Streamlit  

---

# Live Demo

Try the deployed application here:

**Streamlit App:**  
https://credit-card-fraud-detection-athira.streamlit.app/

---

# Problem Statement

Credit card fraud results in substantial financial losses for institutions and customers. Traditional rule-based systems struggle to detect evolving fraud patterns and often produce high false positives.

This project aims to build an intelligent fraud detection system capable of classifying transactions as:

- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**

while improving prediction reliability and minimizing missed fraud cases.

---

# Key Features

✔ Fraud prediction using ensemble learning models  
✔ Behavioral feature engineering for risk detection  
✔ Probability-based risk assessment  
✔ Real-time fraud analysis through Streamlit  
✔ Risk categorization (Low / Medium / High)

---

# Technologies Used

## Programming Language

- Python

## Libraries & Tools

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

# Dataset Features

The model uses transaction-related attributes such as:

| Feature | Description |
|---------|-------------|
| amount | Transaction amount |
| transaction_hour | Hour of transaction |
| merchant_category | Merchant category |
| velocity_last_24h | Number of recent transactions |
| cardholder_age | Age of cardholder |
| foreign_transaction | International transaction indicator |
| location_mismatch | Billing vs transaction location mismatch |
| device_trust_score | Device reliability score |

Target variable:

```text
0 → Legitimate
1 → Fraudulent
```

---

# Machine Learning Workflow

```text
Data Collection
      ↓
Data Preprocessing
      ↓
EDA
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Model Saving (.pkl)
      ↓
Streamlit Deployment
```

---

# Feature Engineering

Additional behavioral features were created to improve fraud detection performance.

### High Risk Transaction

Generated when:

```python
foreign_transaction == 1
AND
location_mismatch == 1
```

### Night Transaction

Generated when:

```python
transaction_hour >= 22
OR transaction_hour <= 4
```

These engineered features help identify suspicious transaction behavior.

---

# Models Evaluated

Multiple ensemble learning models were benchmarked:

- Random Forest
- Gradient Boosting
- XGBoost

Final selected model:

✔ **XGBoost** *(best overall performance)*

Model selection was based on:

- Accuracy
- Recall
- Precision
- F1-score
- ROC-AUC performance

---

# Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix

For fraud detection:

**Recall** and **ROC-AUC** are especially important because missing fraudulent transactions has a high cost.

---

# Streamlit Application

The deployed application collects:

## User Inputs

- Transaction amount
- Merchant category
- Transaction velocity
- Cardholder age
- Transaction hour
- Device trust score
- Foreign transaction status
- Location mismatch status

## Model Outputs

✔ Fraud probability  
✔ Prediction result (Fraud / Legitimate)  
✔ Risk level classification  
✔ Alert messages  
✔ Technical feature vector  

---

## Risk Classification

| Probability | Risk Level |
|-------------|------------|
| ≥ 0.80 | High Risk |
| ≥ 0.40 | Medium Risk |
| < 0.40 | Low Risk |

---

# Run Locally

Clone repository:

```bash
git clone https://github.com/athirasuresh338/credit-card-fraud-detection.git
```

Move into folder:

```bash
cd credit-card-fraud-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Repository Structure

```plaintext
project/
│
├── app.py
├── credit_card.png
├── fraud_detection_model.pkl
├── model_features.pkl
├── credit_card_fraud.csv
├── Credit_Card_Fraud_Ensemble_Project.ipynb
├── requirements.txt
├── README.md
└── screenshot.png
```

---

# Future Improvements

Potential enhancements:

- REST API deployment
- Continuous model retraining
- Explainable AI integration
- Deep learning approaches
- Real-time streaming predictions
- Cloud-based monitoring

---

# Author

**Athira Suresh**

---