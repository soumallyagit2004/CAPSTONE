# RV University - School of Computer Science and Engineering
## CS2235 - Data Science Capstone Project
### Title: Automated Credit Risk and Loan Approval Prediction Using Machine Learning

**Team Members:**
1. **Soumallya Naskar** (USN: `1RUA25SCS0103`)
2. **Sam William B** (USN: `1RUA25SCS0084`)
3. **Shaik Mohammed Naihan** (USN: `1RUA25SCS0091`)

**Program:** B.Sc. (Hons.) Data Science, Semester III  
**Course Code:** CS2235 - Data Science  

---

## 1. Problem Statement
In retail banking, evaluating loan applications is a critical yet complex process because applicants exhibit wide variations in income stability, loan obligations, and repayment behavior. Manual underwriting is time-consuming and prone to subjective inconsistencies when determining whether an applicant is creditworthy or at high risk of default.

**Research Question:** How can we accurately classify loan applications into eligibility outcomes (`Loan_Status`: Approved vs. Rejected) and meaningful credit risk tiers based on applicant credit history, combined household income, and post-EMI residual cashflow using supervised machine learning?

---

## 2. Machine Learning Problem Identification
In machine learning, tasks are divided into three main paradigms:
1. **Supervised Learning - Regression:** Predicts a continuous numerical value (e.g., predicting the exact future price of a house or stock).
2. **Supervised Learning - Classification:** Predicts a predefined categorical label using labeled historical data (e.g., classifying a loan application as Approved or Rejected).
3. **Unsupervised Learning - Clustering:** Discovers natural groupings and hidden patterns in data without predefined labels or target outputs.

**Why Classification for our Project?**
Our historical banking dataset (`imbalanced.csv`) contains predefined, ground-truth outcome labels (`Loan_Status`: `Y` for Approved and `N` for Rejected) recorded from past institutional decisions. Because the target category is known during training, this problem is best solved using **Supervised Learning — Binary Classification (Random Forest Classifier)** to learn decision rules and predict outcomes for new applicants.

---

## 3. Project Workflow & Phases

### Phase 1: Problem Definition & Data Understanding
- Define objectives: Predict loan approval status (`Loan_Status`) and categorize applicant credit risk using demographic, income, and credit history attributes.
- Identify the dataset features in `imbalanced.csv`: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, and `Loan_Status`.

### Phase 2: Data Preprocessing & Exploratory Data Analysis (EDA)
- Inspect dataset shape, structure, and missing/null values (`df.isnull().sum()`).
- Display statistical summaries of numerical features (`df.describe()`).
- Impute missing categorical values using **Mode** and numerical loan amounts using **Median**.
- Treat extreme outliers in income and loan amounts using Interquartile Range (IQR) winsorization ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$).

### Phase 3: Feature Engineering, Encoding & Scaling (StandardScaler)
- Engineer domain-specific financial features:
  - `Total_Income` $= \text{ApplicantIncome} + \text{CoapplicantIncome}$
  - `EMI` $= \frac{\text{LoanAmount} \times 1000}{\text{Loan\_Amount\_Term}}$
  - `Balance_Income` $= \text{Total\_Income} - \text{EMI}$
- Apply logarithmic transformation (`np.log1p`) to normalize skewed financial features and encode categorical columns into numerical format.
- Apply **StandardScaler** to normalize continuous features to mean $\mu = 0$ and variance $\sigma^2 = 1$, ensuring large income numbers do not overshadow smaller numerical features.

### Phase 4: Random Forest Classification & Model Evaluation
- Split the dataset into $80\%$ training and $20\%$ testing sets using stratified sampling (`train_test_split`).
- Determine the optimal maximum tree depth ($d = 2$ to $d = 10$) by plotting validation accuracy across tree depths.
- Train the **Random Forest Classifier** with optimal depth (`RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`).
- Evaluate classification performance using `accuracy_score` and `classification_report` (Precision, Recall, and F1-Score).

### Phase 5: Evaluation Visualizations & Feature Importance Analysis
- Generate a 2D **Confusion Matrix** heatmap (`confusion_matrix`) to visualize true vs. predicted approvals and rejections.
- Plot a horizontal bar chart of **Random Forest Feature Importances** (`model.feature_importances_`) to identify the primary financial drivers of loan approval.

### Phase 6: Risk Tier Interpretation & New Data Prediction
- Analyze predicted approval probabilities and categorize applicants into clear, interpretable underwriting risk tiers:
  - **Tier 1: Low Risk / Prime Applicants** (Clean credit history $1.0$, high residual `Balance_Income`, approval probability $\ge 75\%$ — *Auto-Approve*)
  - **Tier 2: Moderate Risk / Standard Applicants** (Good credit standing, moderate EMI-to-income burden, approval probability $50\%$ to $74.9\%$ — *Standard Underwriting*)
  - **Tier 3: High Risk / Subprime Applicants** (Poor credit history $0.0$ or low residual cashflow, approval probability $< 50\%$ — *Reject / Manual Review*)
- Predict loan eligibility and risk tier for unseen/new applicant data using `live_scaler.transform()` and `model.predict()`.

---

## 4. Expected Deliverables & GitHub Repository Structure
```
CAPSTONE/
│
├── imbalanced.csv                              # Dataset with 472 processed loan applicant records
├── CS2235_Loan_Approval_Prediction.ipynb       # Complete Google Colab / Jupyter Notebook (Phases 1-6)
├── Project_Proposal.md                         # Capstone project proposal with problem definition & workflow
├── Project_Report.md                           # Simple, student-friendly academic report
└── README.md                                   # Step-by-step project guide & Colab instructions
```

---

## 5. Technology Stack (Beginner Level)
- **Language:** Python 3
- **Libraries:**
  - `pandas` - Data loading, cleaning, imputation, and table manipulation
  - `numpy` - Numerical calculations, IQR clipping, and logarithmic transformations
  - `matplotlib.pyplot` & `seaborn` - Accuracy curves, Confusion Matrix, and Feature Importance plots
  - `sklearn.preprocessing.StandardScaler` - Feature normalization
  - `sklearn.model_selection.train_test_split` - Stratified train-test data splitting
  - `sklearn.ensemble.RandomForestClassifier` - Supervised classification algorithm
  - `sklearn.metrics` - Accuracy score, Classification Report, and Confusion Matrix validation

---

## 6. Team Division of Work

| Sl. No. | Student Name | USN | Assigned Workload / Phases |
| :---: | :--- | :---: | :--- |
| 1 | **Soumallya Naskar** | `1RUA25SCS0103` | Problem Definition & ML Identification (Phase 1), Random Forest Model Training (Phase 4), New Data Prediction (Phase 6), Proposal & Report writing |
| 2 | **Sam William B** | `1RUA25SCS0084` | Dataset Curation (`imbalanced.csv`), Missing Value Imputation, IQR Outlier Treatment & Exploratory Data Analysis (Phase 2), Notebook testing |
| 3 | **Shaik Mohammed Naihan** | `1RUA25SCS0091` | Domain Feature Engineering & StandardScaler Normalization (Phase 3), Confusion Matrix & Feature Importance Visualizations (Phase 5) |