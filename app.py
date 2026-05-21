import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Load Model and Features
@st.cache_resource
def load_model_assets():
    model = joblib.load("fraud_detection_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features


try:
    model, model_features = load_model_assets()

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# Header
st.title("💳 Credit Card Fraud Detection")

st.markdown(
    "### Detect suspicious transactions instantly using machine learning-powered fraud analysis."
)

st.image("credit_card.png", use_container_width=True)

st.markdown("**Enter transaction details below to evaluate fraud risk.**")

# Input Section
col1, col2 = st.columns(2)

with col1:

    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=100.0,
        step=10.0,
        help="Total monetary value of the transaction."
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "Electronics",
            "Food",
            "Grocery",
            "Travel"
        ],
        help="Category representing the type of merchant involved in the transaction."
    )

    velocity_last_24h = st.number_input(
        "Transaction Velocity (Last 24h)",
        min_value=0,
        value=1,
        help="Number of transactions performed by the cardholder within the last 24 hours."
    )

    cardholder_age = st.slider(
        "Cardholder Age",
        18,
        100,
        30,
        help="Age of the credit card holder."
    )

with col2:

    transaction_hour = st.slider(
        "Hour of Day (0-23)",
        0,
        23,
        12,
        help="Hour at which the transaction was performed."
    )

    device_trust_score = st.slider(
        "Device Trust Score (0-100)",
        0,
        100,
        85,
        help="Trustworthiness score assigned to the device used for the transaction."
    )

    sub_col1, sub_col2 = st.columns(2)

    with sub_col1:

        foreign_transaction = st.radio(
            "Foreign Transaction?",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            help="Indicates whether the transaction was performed internationally."
        )

    with sub_col2:

        location_mismatch = st.radio(
            "Location Mismatch?",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            help="Indicates whether the transaction location differs from the billing location."
        )


# Typical ranges information
st.info("""
Typical ranges commonly observed during model training:

• Transaction Amount: 0–1500  
• Transaction Velocity: 0–9  
• Device Trust Score: 25–100
""")


# Feature Engineering
high_risk_transaction = int(
    foreign_transaction == 1 and
    location_mismatch == 1
)

night_transaction = int(
    transaction_hour >= 22 or
    transaction_hour <= 4
)

# Prediction
if st.button(
    "Run Fraud Analysis",
    type="primary",
    use_container_width=True
):

    # Input Data
    input_data = pd.DataFrame({
        'amount': [amount],
        'transaction_hour': [transaction_hour],
        'foreign_transaction': [foreign_transaction],
        'location_mismatch': [location_mismatch],
        'device_trust_score': [device_trust_score],
        'velocity_last_24h': [velocity_last_24h],
        'cardholder_age': [cardholder_age],
        'high_risk_transaction': [high_risk_transaction],
        'night_transaction': [night_transaction],
        'merchant_category': [merchant_category]
    })

    # Encoding and Alignment
    input_encoded = pd.get_dummies(input_data)

    input_final = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_final)[0]

    probability = model.predict_proba(
        input_final
    )[0][1]

    # Risk Level
    if probability >= 0.7:
        risk_level = "High"

    elif probability >= 0.2:
        risk_level = "Medium"

    else:
        risk_level = "Low"


    # Results
    st.subheader("Analysis Results")

    out_of_distribution_inputs = []

    if amount > 1500:
        out_of_distribution_inputs.append(
            "Transaction Amount"
        )

    if velocity_last_24h > 9:
        out_of_distribution_inputs.append(
            "Transaction Velocity"
        )

    if device_trust_score < 25:
        out_of_distribution_inputs.append(
            "Device Trust Score"
        )


    if out_of_distribution_inputs:

        inputs_text = ", ".join(
            out_of_distribution_inputs
        )

        st.caption(
            f"Note: {inputs_text} differ from values commonly observed during model training."
        )

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:

        st.metric(
            label="Fraud Probability",
            value=f"{probability:.2%}"
        )

        st.caption(
            "Higher probability indicates stronger similarity "
            "to fraud patterns learned from the training dataset."
        )

    with res_col2:

        status = (
            "🔴 FRAUD"
            if prediction == 1
            else "🟢 LEGIT"
        )

        st.metric(
            label="Model Decision",
            value=status
        )

    with res_col3:

        st.metric(
            label="Risk Level",
            value=risk_level
        )


    # Alerts
    if prediction == 1:

        st.error(
            "🚨 High Risk Alert: "
            "This transaction matches "
            "known fraud patterns."
        )

    elif probability >= 0.4:

        st.warning(
            "⚠️ Moderate Risk: "
            "Transaction contains "
            "suspicious elements."
        )

    else:

        st.success(
            "✅ Legitimate Transaction"
        )


    st.warning(
        "⚠️ Disclaimer: This model was trained on a synthetic fraud dataset. "
        "Predictions reflect learned patterns within generated data and may not "
        "fully represent real-world fraud behavior."
    )

    # Technical Feature View
    with st.expander(
        "View Technical Feature Vector"
    ):

        st.dataframe(input_final)