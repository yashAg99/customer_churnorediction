# app.py

import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('customer_churn_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("📊 Customer Churn Prediction App")


# User inputs
gender = st.selectbox("Gender", ["Female", "Male"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)

# Encode and scale input
def encode_input(gender, tenure, contract, monthlycharges):
    gender_map = {"Female": 0, "Male": 1}
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

    data = [gender_map[gender], tenure, contract_map[contract], monthlycharges]
    data[3] = scaler.transform([[data[3]]])[0][0]  # Scale MonthlyCharges only
    return np.array([data])

# Predict
if st.button("Predict"):
    features = encode_input(gender, tenure, contract, monthlycharges)
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.warning("🟡 This customer is likely to **churn**.")
    else:
        st.success("🟢 This customer is likely to **stay**.")
