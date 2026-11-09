# ==========================================
# CS2235 Capstone Project - Model Training & Export
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load engineered dataset
df = pd.read_csv("C:\\Users\\naska\\OneDrive\\Desktop\\capstone\\loan_data_etl_fe.csv")
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Train-test split (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model artifacts
joblib.dump(model, "best_loan_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")

# Fit scaler on raw engineered training data for live app inputs
file_path = r'C:\Users\naska\OneDrive\Desktop\capstone\train.csv.csv'
if not os.path.exists(file_path):
    file_path = 'train.csv.csv'

raw_df = pd.read_csv('C:\\Users\\naska\\OneDrive\\Desktop\\capstone\\loan_data_cleaned.csv')
if "Loan_ID" in raw_df.columns:
    raw_df.drop(columns=["Loan_ID"], inplace=True)
raw_df.drop_duplicates(inplace=True)
if "Dependents" in raw_df.columns:
    raw_df["Dependents"] = raw_df["Dependents"].replace("3+", "3")

mode_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]
for col in mode_cols:
    if col in raw_df.columns:
        raw_df[col] = raw_df[col].fillna(raw_df[col].mode()[0])

raw_df["Dependents"] = raw_df["Dependents"].astype(int)
raw_df["LoanAmount"] = raw_df["LoanAmount"].fillna(raw_df["LoanAmount"].median())

raw_df["Total_Income"] = raw_df["ApplicantIncome"] + raw_df["CoapplicantIncome"]
raw_df["EMI"] = (raw_df["LoanAmount"] * 1000) / raw_df["Loan_Amount_Term"]
raw_df["Balance_Income"] = raw_df["Total_Income"] - raw_df["EMI"]
raw_df["Log_Total_Income"] = np.log1p(raw_df["Total_Income"])
raw_df["Log_LoanAmount"] = np.log1p(raw_df["LoanAmount"])

num_cols_to_scale = ["Dependents", "Loan_Amount_Term", "EMI", "Balance_Income", "Log_Total_Income", "Log_LoanAmount"]
live_scaler = StandardScaler()
live_scaler.fit(raw_df[num_cols_to_scale])
joblib.dump(live_scaler, "loan_scaler.pkl")

print("\nModel training complete! Artifacts saved successfully.")