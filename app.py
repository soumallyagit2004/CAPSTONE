# ==========================================
# CS2235 Capstone Project - Streamlit Web App (Professional UI)
# Run command in terminal: streamlit run app.py
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk & Loan Approval Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_loan_model.pkl")
    scaler = joblib.load("loan_scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

st.title("Automated Loan Approval Prediction System")
st.markdown("### RV University | CS2235 Data Science Capstone Project")
st.write("Enter the applicant's official financial, demographic, and credit history details below to evaluate retail loan eligibility.")

with st.form("loan_application_form"):
    st.subheader("Applicant Profile & Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"], format_func=lambda x: "Married" if x == "Yes" else "Unmarried")
        dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3], format_func=lambda x: f"{x} Dependent(s)" if x > 0 else "None (0)")
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Employment Type (Self-Employed?)", ["No", "Yes"], format_func=lambda x: "Salaried / Employed" if x == "No" else "Self-Employed")
        
    with col2:
        credit_history = st.selectbox(
            "Credit History (Repayment Guidelines Met?)",
            options=[1.0, 0.0],
            format_func=lambda x: "Yes (1.0 - Good Standing / Clear History)" if x == 1.0 else "No (0.0 - Past Defaults / No History)"
        )
        property_area = st.selectbox("Property Location Type", ["Semiurban", "Urban", "Rural"])
        
    st.subheader("Financial & Loan Details")
    col3, col4 = st.columns(2)
    
    with col3:
        applicant_income = st.number_input("Primary Applicant Monthly Income (₹)", min_value=1000, value=50000, step=1000)
        coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, value=15000, step=1000)
        
    with col4:
        loan_amount = st.number_input("Requested Loan Amount (in Thousands ₹, e.g., 150 = ₹1,50,000)", min_value=10.0, value=150.0, step=5.0)
        loan_term = st.selectbox(
            "Loan Repayment Tenure", 
            options=[360.0, 180.0, 480.0, 300.0, 240.0, 120.0],
            format_func=lambda x: f"{int(x)} Months ({int(x)//12} Years)"
        )

    submitted = st.form_submit_button("Evaluate Loan Eligibility")

if submitted:
    total_income = applicant_income + coapplicant_income
    emi = (loan_amount * 1000.0) / loan_term
    balance_income = total_income - emi
    log_total_income = np.log1p(total_income)
    log_loan_amount = np.log1p(loan_amount)
    
    input_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": float(credit_history),
        "EMI": emi,
        "Balance_Income": balance_income,
        "Log_Total_Income": log_total_income,
        "Log_LoanAmount": log_loan_amount,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0
    }
    
    input_df = pd.DataFrame([input_data])[feature_names]
    
    num_cols_to_scale = ["Dependents", "Loan_Amount_Term", "EMI", "Balance_Income", "Log_Total_Income", "Log_LoanAmount"]
    input_df[num_cols_to_scale] = scaler.transform(input_df[num_cols_to_scale])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    st.subheader("Underwriting Evaluation Result")
    if prediction == 1:
        st.success(f"✅ **LOAN APPROVED** — The applicant meets the institutional risk criteria. (Confidence Score: {probability * 100:.1f}%)")
    else:
        st.error(f"❌ **LOAN REJECTED** — High credit risk or insufficient residual balance income. (Approval Probability: {probability * 100:.1f}%)")
        
    with st.expander("View Automated Financial Calculations & Diagnostics"):
        st.write(f"- **Total Household Monthly Income:** ₹{total_income:,.2f}")
        st.write(f"- **Estimated Monthly Installment (EMI):** ₹{emi:,.2f} / month")
        st.write(f"- **Residual Balance Income (Post-EMI Cashflow):** ₹{balance_income:,.2f}")