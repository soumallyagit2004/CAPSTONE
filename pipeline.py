# ==========================================
# CS2235 Capstone Project - Phases 2 & 3 Pipeline
# ==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("1. Loading dataset...")
df = pd.read_csv('C:\\Users\\naska\\OneDrive\\Desktop\\capstone\\imbalanced_dataset.csv')
print("Initial Shape:", df.shape)

# --- PHASE 2: DATA CLEANING ---
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)
df.drop_duplicates(inplace=True)

if "Dependents" in df.columns:
    df["Dependents"] = df["Dependents"].replace("3+", "3")

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()

# Missing value imputation (mode)
mode_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]
for col in mode_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df["Dependents"] = df["Dependents"].replace('nan', np.nan)
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0]).astype(int)

if "LoanAmount" in df.columns:
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

# Outlier Winsorization
for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

df.to_csv("loan_data_cleaned.csv", index=False)
print("Phase 2 Complete! Cleaned Shape:", df.shape)


# --- PHASE 3: ETL & FEATURE ENGINEERING ---
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["EMI"] = (df["LoanAmount"] * 1000) / df["Loan_Amount_Term"]
df["Balance_Income"] = df["Total_Income"] - df["EMI"]

df["Log_Total_Income"] = np.log1p(df["Total_Income"])
df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])

binary_maps = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Loan_Status": {"Y": 1, "N": 0}
}
for col, mapping in binary_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True, dtype=int)

df_fe = df.drop(columns=["ApplicantIncome", "CoapplicantIncome", "Total_Income", "LoanAmount"], errors='ignore')
X = df_fe.drop(columns=["Loan_Status"])
y = df_fe["Loan_Status"]

num_cols = ["Dependents", "Loan_Amount_Term", "EMI", "Balance_Income", "Log_Total_Income", "Log_LoanAmount"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

final_df = pd.concat([X, y], axis=1)
final_df.to_csv("loan_data_etl_fe.csv", index=False)
print("Phase 3 Complete! Final Processed Shape:", final_df.shape)