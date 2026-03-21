#  Credit Risk Prediction
### Binary Classification · Logistic Regression · Model Evaluation · EDA

## Overview

A complete **binary classification project** that predicts whether a loan applicant is likely to default on a loan. The project covers the full machine learning pipeline — from raw data loading and missing value treatment, through exploratory data analysis, to model training and evaluation using Logistic Regression.

Built entirely in a **Jupyter Notebook inside VS Code**, with every section documented using markdown for clarity and reproducibility.

---

## The Dataset

| Property       | Detail                                         |
|----------------|------------------------------------------------|
| Source         | Kaggle — Loan Prediction Problem Dataset       |
| Total Samples  | 614 applicants                                 |
| Features       | 11 input features (numerical + categorical)    |
| Target Column  | `Loan_Status` — Y (Approved) / N (Rejected)   |
| Missing Values | Present in 6 columns — handled via imputation  |

### Features

| Feature             | Type        | Description                                 |
|---------------------|-------------|---------------------------------------------|
| Gender              | Categorical | Male / Female                               |
| Married             | Categorical | Applicant marital status                    |
| Dependents          | Categorical | Number of dependents                        |
| Education           | Categorical | Graduate / Not Graduate                     |
| Self_Employed       | Categorical | Whether applicant is self-employed          |
| ApplicantIncome     | Numerical   | Monthly income of applicant                 |
| CoapplicantIncome   | Numerical   | Monthly income of co-applicant              |
| LoanAmount          | Numerical   | Loan amount requested (in thousands)        |
| Loan_Amount_Term    | Numerical   | Repayment term in months                    |
| Credit_History      | Numerical   | 1 = good credit history, 0 = bad            |
| Property_Area       | Categorical | Urban / Semiurban / Rural                   |

### Target Variable

| Value | Meaning            |
|-------|--------------------|
| `Y`   | Loan Approved      |
| `N`   | Loan Rejected (default risk) |

## Stack

| Tool              | Role                                              |
|-------------------|---------------------------------------------------|
| `pandas`          | Data loading, cleaning, grouping, and summaries   |
| `numpy`           | Numerical operations                              |
| `matplotlib`      | Base chart rendering and plot layouts             |
| `seaborn`         | Statistical visualizations — heatmap, box plots   |
| `scikit-learn`    | Model training, encoding, splitting, evaluation   |


## Project Structure

```
credit-risk-prediction/
│
├── credit_risk_prediction.ipynb   ← Full notebook with markdown + code
├── train.csv                      ← Dataset from Kaggle (place here)
└── README.md                      ← This file
```

## Setup

**Clone and install:**

```bash
git clone https://github.com/yourusername/credit-risk-prediction.git
cd credit-risk-prediction
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Download the dataset:**

👉 [Kaggle — Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

Place `train.csv` in the same folder as the notebook before running.

**Run the notebook:**

Open `credit_risk_prediction.ipynb` in VS Code, select your Python interpreter, and run all cells with `Shift + Enter`.

> Requires the **Jupyter extension** in VS Code.

---

## Analysis Breakdown

### Missing Value Treatment
Six columns contained missing data. Numerical columns (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`) were imputed using the **median** — chosen over mean due to right-skewed income distributions where outliers would distort the average. Categorical columns (`Gender`, `Married`, `Dependents`, `Self_Employed`) were filled using the **mode** (most frequent value), preserving the natural distribution of each category.

### Exploratory Data Analysis

**Target Distribution** — The dataset is moderately imbalanced: approximately 69% of applicants were approved and 31% were rejected. This imbalance is noted and considered when interpreting accuracy scores.

**Income and Loan Amount** — Applicant income is heavily right-skewed, with most applicants earning moderate amounts and a small number of high earners pulling the distribution tail. Loan amounts follow a similar pattern. Overlapping histograms by loan status reveal that income alone is not a strong separator between approved and rejected applicants.

**Categorical Features** — Grouped bar charts show that graduates have a noticeably higher approval rate than non-graduates. Self-employment shows minimal impact on approval rates. Credit history is the most visually striking feature — applicants with no credit history (value = 0) are rejected at a dramatically higher rate.

**Loan Amount by Group** — Box plots across education level and property area show that graduates and urban applicants tend to request larger loan amounts, with more high-value outliers in both groups.

**Correlation Heatmap** — Among numerical features, `Credit_History` has the strongest positive correlation with `Loan_Status` (≈ 0.54). Income features show weak individual correlations with the target, suggesting that categorical features carry significant predictive weight in this dataset.

### Feature Encoding
All categorical columns were converted to numerical values using `LabelEncoder` before model training. `Loan_ID` was dropped as it carries no predictive information. After encoding, the dataset contained only integer and float columns — fully compatible with Logistic Regression.

### Model — Logistic Regression
Logistic Regression was selected as the classification model due to its interpretability and strong baseline performance on tabular binary classification tasks. The dataset was split 80% training and 20% testing with `stratify=y` to preserve the class ratio in both sets. `max_iter=1000` was set to ensure full convergence on this dataset.

### Model Evaluation
Performance was assessed using four metrics to avoid over-reliance on accuracy alone:

| Metric    | Description                                                    |
|-----------|----------------------------------------------------------------|
| Accuracy  | Overall percentage of correct predictions                      |
| Precision | Of all predicted approvals, how many were genuinely approved?  |
| Recall    | Of all actual approvals, how many did the model catch?         |
| F1 Score  | Harmonic mean of precision and recall                          |

The confusion matrix provides the error breakdown — identifying both **False Positives** (creditworthy applicants wrongly rejected) and **False Negatives** (risky applicants wrongly approved), which have different real-world costs in a lending context.

---

## Key Findings

**1. Credit History is the dominant predictor.**
Applicants with a bad credit history (value = 0) are rejected at a disproportionately high rate across all other feature combinations. It is the single most informative feature in the dataset.

**2. Income is less decisive than expected.**
Despite being a central factor in real-world lending, applicant income showed weak direct correlation with loan approval in this dataset. The combination of income with loan amount (debt-to-income ratio) is more informative than either feature alone.

**3. Graduates are approved more frequently.**
Education level shows a meaningful association with approval rates — graduates represent the majority of approved applicants, likely reflecting more stable and verifiable income sources.

**4. The dataset is moderately imbalanced.**
With approximately 69% approvals, a naive classifier that always predicts approval would achieve ~69% accuracy. The model must be evaluated on F1 Score and the confusion matrix to confirm it is genuinely learning — not just predicting the majority class.

**5. Property area has limited discriminating power.**
Urban, Semiurban, and Rural applicants show broadly similar approval rates, making this a low-importance feature relative to credit history and education.


## Statistical Summary

| Feature             | Mean     | Std Dev  | Min    | Max      |
|---------------------|----------|----------|--------|----------|
| ApplicantIncome     | 5403.46  | 6109.04  | 150    | 81000    |
| CoapplicantIncome   | 1621.25  | 2926.25  | 0      | 41667    |
| LoanAmount          | 146.41   | 85.59    | 9      | 700      |
| Loan_Amount_Term    | 342.00   | 65.12    | 12     | 480      |
| Credit_History      | 0.84     | 0.36     | 0      | 1        |

---

## Author

**Momina Ramzan**

## Acknowledgements

- [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) — dataset source
- Scikit-learn contributors — model and evaluation tools
- UCI Machine Learning Repository — original dataset reference
