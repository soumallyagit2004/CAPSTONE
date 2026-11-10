# Automated Credit Risk and Loan Approval Prediction Using Machine Learning

**Course:** CS2235 - Data Science Capstone Project  
**Program:** B.Sc. (Hons.) Data Science, Semester III  
**Institution:** School of Computer Science and Engineering, RV University, Bengaluru  

### Team Members:
1. **Soumallya Naskar** — USN: `1RUA25SCS0103`
2. **Sam William B** — USN: `1RUA25SCS0084`
3. **Shaik Mohammed Naihan** — USN: `1RUA25SCS0091`  

---

## 1. Introduction & Problem Statement
Retail loan underwriting is a core financial operation in modern banking. However, loan applicants exhibit significant differences in financial stability and credit behavior:
- Prime applicants maintain a clean credit repayment record and possess strong post-installment residual household income.
- Moderate-risk applicants have stable incomes but higher monthly debt obligations relative to their earnings.
- High-risk applicants have prior credit defaults or insufficient disposable cashflow to cover monthly loan installments.

### Objective
To analyze historical banking applicant data from `imbalanced.csv` and classify loan applications into eligibility outcomes (`Loan_Status`: Approved vs. Rejected) and credit risk tiers based on:
1. **Credit History & Repayment Standing**: Whether the applicant has met past institutional debt obligations (`Credit_History`).
2. **Household Cashflow & Repayment Capacity**: Combined household monthly income (`Total_Income`), Estimated Monthly Installment (`EMI`), and post-EMI residual balance (`Balance_Income`).

---

## 2. Machine Learning Problem Identification
A primary conceptual requirement in Data Science is choosing the right machine learning paradigm:
- **Regression:** Predicts a continuous numeric outcome (e.g., predicting exact house prices or future stock values).
- **Classification (Supervised Learning):** Predicts discrete categories using pre-labeled training data (e.g., classifying a loan application as Approved or Rejected).
- **Clustering (Unsupervised Learning):** Groups similar data points based on feature similarity without predefined labels.

**Why Classification?**  
In our historical banking dataset (`imbalanced.csv`), every past applicant record already contains an official, ground-truth outcome label (`Loan_Status`: `Y` for Approved and `N` for Rejected). Therefore, we train a **Supervised Machine Learning Classifier (Random Forest)** to learn the decision boundaries from historical data and predict outcomes for new, unseen applicants.

---

## 3. Dataset Description
The project uses the dataset `imbalanced.csv` comprising $472$ processed applicant records across demographic, financial, and property attributes.

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Gender` | Categorical | Gender of primary applicant (`Male` / `Female`) |
| `Married` | Categorical | Marital status of applicant (`Yes` / `No`) |
| `Dependents` | Numerical (Int) | Number of dependents supported by the applicant ($0, 1, 2, 3+$) |
| `Education` | Categorical | Education qualification (`Graduate` / `Not Graduate`) |
| `Self_Employed` | Categorical | Employment category (`Yes` / `No`) |
| `ApplicantIncome` | Numerical (Float) | Monthly income of the primary applicant |
| `CoapplicantIncome` | Numerical (Float) | Monthly income of the co-applicant |
| `LoanAmount` | Numerical (Float) | Requested loan principal amount (in thousands) |
| `Loan_Amount_Term` | Numerical (Float) | Loan repayment tenure in months (e.g., $360$ months) |
| `Credit_History` | Numerical (Binary) | Past credit repayment record ($1.0 =$ Good, $0.0 =$ Poor) |
| `Property_Area` | Categorical | Property classification (`Urban`, `Semiurban`, `Rural`) |
| `Loan_Status` | Categorical (Target) | Target approval status (`Y` = Approved, `N` = Rejected) |

---

## 4. Methodology & Implementation Phases

### Phase 1: Data Understanding & Preprocessing
- Loaded data from `imbalanced.csv` using `pandas.read_csv()`.
- Removed identifier columns (`Loan_ID`) and duplicate records, and checked missing values using `df.isnull().sum()`.
- Imputed missing categorical values using statistical **Mode** and missing `LoanAmount` values using **Median**.
- Capped extreme numerical outliers in income and loan amount columns using Interquartile Range (IQR) winsorization ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$).

### Phase 2: Feature Engineering, Encoding & Scaling
- Engineered three domain-specific financial indicators:
  $$\text{Total\_Income} = \text{ApplicantIncome} + \text{CoapplicantIncome}$$
  $$\text{EMI} = \frac{\text{LoanAmount} \times 1000}{\text{Loan\_Amount\_Term}}$$
  $$\text{Balance\_Income} = \text{Total\_Income} - \text{EMI}$$
- Applied logarithmic transformation ($\ln(1 + x)$ via `np.log1p`) on skewed income and loan amount variables, and encoded categorical attributes into numerical values.
- **StandardScaler Normalization:** Because household incomes and balances are in thousands while dependents and terms operate on different scales, we applied `StandardScaler()` so each numerical feature has $\mu = 0$ and $\sigma^2 = 1$.

### Phase 3: Determining Optimal Model Complexity (Tree Depth Evaluation)
- Partitioned the dataset into $80\%$ training and $20\%$ testing sets using stratified sampling (`train_test_split` with `stratify=y`).
- Evaluated Random Forest test accuracy across maximum tree depths from $d = 2$ to $d = 10$ to prevent overfitting and select an optimal tree depth (`max_depth=6`).

### Phase 4: Random Forest Classification & Evaluation
- Trained `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`.
- Evaluated classification quality using **Test Accuracy**, **Precision**, **Recall**, and **F1-Score**:
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  The model achieved an overall test accuracy of $89.47\%$ ($0.8947$), with an F1-Score of $0.94$ on approved loans.

### Phase 5: Evaluation Visualizations & Risk Interpretation
- Generated a **Confusion Matrix** heatmap to inspect true positives, true negatives, false positives, and false negatives on the test set.
- Plotted a horizontal bar chart of **Random Forest Feature Importances**, confirming the top predictors driving loan decisions.

### Phase 6: New Data Prediction
- Used `live_scaler.transform()` and `model.predict()` / `model.predict_proba()` directly in the notebook to predict the loan eligibility status and underwriting risk tier for new, unseen applicant profiles.

---

## 5. Results & Discussion

The Random Forest model achieved **$89.47\%$ test accuracy** and enabled categorization of applicants into $3$ interpretable underwriting risk tiers based on predicted approval probabilities:

| Risk Tier | Characteristics | Underwriting Action |
| :--- | :--- | :--- |
| **Tier 1: Low Risk (Prime)** | Clean credit history ($1.0$), high residual `Balance_Income`, approval probability $\ge 75\%$ | Auto-Approve |
| **Tier 2: Moderate Risk (Standard)** | Satisfactory credit standing, moderate EMI-to-income ratio, approval probability $50\%$ to $74.9\%$ | Standard Underwriting |
| **Tier 3: High Risk (Subprime)** | Poor credit history ($0.0$) or low post-EMI residual income, approval probability $< 50\%$ | Reject / Manual Review |

---

## 6. Conclusion
- The Random Forest classification pipeline accurately separated creditworthy loan applicants from high-risk profiles with $89.47\%$ accuracy on `imbalanced.csv`.
- Domain-engineered features (`Balance_Income`, `Log_Total_Income`, and `EMI`) alongside `Credit_History` proved to be the strongest predictors of loan approval.
- The notebook pipeline is modular, interpretable, and reproducible on Google Colab and VS Code Jupyter Notebook.

---

## 7. Team Contributions

| Sl. No. | Student Name | USN | Contribution / Phase Responsibilities |
| :---: | :--- | :---: | :--- |
| 1 | **Soumallya Naskar** | `1RUA25SCS0103` | Problem Definition & ML Formulation (Phase 1), Random Forest Model Training (Phase 4), New Applicant Prediction (Phase 6) & Final Report |
| 2 | **Sam William B** | `1RUA25SCS0084` | Dataset Curation (`imbalanced.csv`), Missing Value Imputation, IQR Outlier Treatment & Exploratory Data Analysis (Phase 2) |
| 3 | **Shaik Mohammed Naihan** | `1RUA25SCS0091` | Domain Feature Engineering, StandardScaler Normalization (Phase 3), Confusion Matrix & Feature Importance Visualizations (Phase 5) |

---

## 8. References
1. RV University CS2235 Data Science Laboratory Manual (Data Preprocessing, Feature Scaling, and Supervised Classification).
2. Scikit-learn Documentation: `sklearn.ensemble.RandomForestClassifier` and `sklearn.preprocessing.StandardScaler`.
3. McKinney, Wes. *Python for Data Analysis*, O'Reilly Media.