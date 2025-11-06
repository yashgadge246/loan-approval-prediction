
# Loan Approval Prediction Web App using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import pickle


try:
    model = pickle.load(open("loan_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    st.success(" Model and Scaler Loaded Successfully!")
except Exception as e:
    st.error(" Error loading model/scaler. Make sure 'loan_model.pkl' and 'scaler.pkl' exist in the same folder.")
    st.stop()


st.title("🏦 Loan Approval Prediction App")
st.markdown("""
This app predicts whether a **loan application will be approved or rejected**  
based on applicant details such as income, assets, and credit score.
""")


st.sidebar.header("🧾 Enter Applicant Details")

def user_input_features():
    no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.sidebar.number_input("Annual Income (₹)", min_value=0, step=10000)
    loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0, step=10000)
    loan_term = st.sidebar.selectbox("Loan Term (Months)", [12, 24, 36, 60, 120, 180, 240, 300, 360])
    cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 750)
    residential_assets_value = st.sidebar.number_input("Residential Asset Value (₹)", min_value=0, step=50000)
    commercial_assets_value = st.sidebar.number_input("Commercial Asset Value (₹)", min_value=0, step=50000)
    luxury_assets_value = st.sidebar.number_input("Luxury Asset Value (₹)", min_value=0, step=50000)
    bank_asset_value = st.sidebar.number_input("Bank Asset Value (₹)", min_value=0, step=50000)

    
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    
    data = {
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


st.subheader("📋 Entered Details:")
st.write(input_df)


if st.button("🔍 Predict Loan Status"):
    
    scaled_input = scaler.transform(input_df)
    
    
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]

    
    st.subheader("📊 Prediction Result:")
    if prediction[0] == 1:
        st.success(f" Loan Approved! (Confidence: {probability*100:.2f}%)")
    else:
        st.error(f" Loan Rejected. (Confidence: {(1 - probability)*100:.2f}%)")


st.markdown("""
---
Made with ❤️ by **Yash**  
Using Logistic Regression and Streamlit
""")
