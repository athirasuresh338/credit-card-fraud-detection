# Credit Card Fraud Detection System

**Python | Streamlit | Scikit-Learn | XGBoost | Machine Learning**

🔗 **Live Demo:** [Try the Dashboard Here](https://credit-card-fraud-detection-athira.streamlit.app/)

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions using **ensemble learning techniques** and provide **real-time fraud risk analysis** through an interactive Streamlit application.

The repository includes the **complete ML workflow**, from data preprocessing and feature engineering to model training, evaluation, and deployment.

---

# Project Overview

Credit card fraud causes significant financial losses for institutions and customers. Traditional rule-based systems often struggle to detect evolving fraud patterns and may generate high false positives.

This project builds an intelligent fraud detection system capable of classifying transactions as:

* **0 → Legitimate Transaction**
* **1 → Fraudulent Transaction**

The system prioritizes:

* Improving prediction reliability
* Minimizing missed fraud cases
* Providing interpretable risk assessment

---

# Key Features

* Fraud prediction using ensemble learning models
* Behavioral feature engineering for risk detection
* Probability-based risk assessment
* Real-time fraud analysis through Streamlit
* Risk categorization (Low / Medium / High)
* Interactive dashboard for instant predictions

---

# Core Stack & Tools

* **Programming Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning Frameworks:** Scikit-learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment & User Interface:** Streamlit
* **Development Environment:** Jupyter Notebook, VS Code

---

# Dataset Features

The model evaluates transaction-related features such as:

| Feature | Description |
| --- | --- |
| `amount` | Transaction amount |
| `transaction_hour` | Hour of transaction |
| `merchant_category` | Merchant category |
| `velocity_last_24h` | Number of recent transactions |
| `cardholder_age` | Cardholder age |
| `foreign_transaction` | International transaction indicator |
| `location_mismatch` | Billing and transaction location mismatch |
| `device_trust_score` | Device reliability score |

Target variable:

```text
0 → Legitimate Transaction
1 → Fraudulent Transaction

```

---

# Machine Learning Workflow

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis (EDA)
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

To capture multi-dimensional fraud signatures, the application synthesizes advanced features from basic input telemetry before model inference:

### 1. Composite High-Risk Event (`high_risk_transaction`)

Flags concurrent out-of-country transactions that simultaneously trigger a billing address mismatch:


$$\text{high\_risk\_transaction} = \begin{cases} 1 & \text{if } \text{foreign\_transaction} = 1 \text{ AND } \text{location\_mismatch} = 1 \\ 0 & \text{otherwise} \end{cases}$$

### 2. High-Risk Operational Window (`night_transaction`)

Isolates transaction events taking place during late-night hours when fraudulent density historically spikes:


$$\text{night\_transaction} = \begin{cases} 1 & \text{if } \text{transaction\_hour} \ge 22 \text{ OR } \text{transaction\_hour} \le 4 \\ 0 & \text{otherwise} \end{cases}$$

---

# Model Evaluation & Optimization

Multiple ensemble paradigms were rigorously benchmarked to identify optimal detection boundaries:

* **Random Forest Classifier** (Baseline Bagging Paradigm)
* **Gradient Boosting Machine (GBM)** (Sequential Boosting)
* **XGBoost** (Optimized Extreme Gradient Boosting)

### Final Selection Verdict: XGBoost

XGBoost delivered the strongest overall performance metrics. Optimization hyper-parameters focused heavily on maximizing **Recall** and **ROC-AUC** scores to keep the system highly sensitive to missed fraud patterns.

---

# Risk Classification Engine

The deployment pipeline passes predictions through an interpretable risk matrix based on continuous classification probability ($\mathbb{P}$):

| Fraud Probability Range ($\mathbb{P}$) | Risk Level Assignment | UI Visual Alert | System Actions |
| --- | --- | --- | --- |
| $\mathbb{P} \ge 80\%$ | High Risk | FRAUD | Deny authorization; trigger fraud review pipeline. |
| $40\% \le \mathbb{P} < 80\%$ | Medium Risk | Moderate Risk | Enforce secondary step-up authentication. |
| $\mathbb{P} < 40\%$ | Low Risk | LEGIT | Authorize and settle transaction seamlessly. |

---

# Repository Layout

```plaintext
├── app.py                             # Streamlit serving application
├── credit_card.png                    # Interface header banner graphics asset
├── fraud_detection_model.pkl          # Serialized production XGBoost model
├── model_features.pkl                 # Reference feature template array
├── credit_card_fraud.csv              # Source evaluation dataset
├── Credit_Card_Fraud_Ensemble_Project.ipynb # Optimization & training notebook
├── requirements.txt                   # Environment package dependencies
└── README.md                          # Documentation management artifact

```

---

# Local Deployment Setup

To spin up your local instance of the analysis engine, execute the following commands in your terminal:

```bash
# 1. Clone the project architecture
git clone https://github.com/athirasuresh338/credit-card-fraud-detection.git

# 2. Access the repository directory
cd credit-card-fraud-detection

# 3. Synchronize library environments
pip install -r requirements.txt

# 4. Initialize the Streamlit service instance
streamlit run app.py

```