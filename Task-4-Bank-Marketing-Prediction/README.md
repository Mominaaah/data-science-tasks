# Task 4 — Term Deposit Subscription Prediction

Predict whether a bank customer will subscribe to a term deposit as a result of a direct marketing campaign using the UCI Bank Marketing Dataset.

## Dataset

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Rows:** 45,211 customer records
- **Features:** 16 input features (demographic + campaign data)
- **Target:** `y` — did the client subscribe to a term deposit? (`yes` / `no`)
- **Class imbalance:** 88.3% No · 11.7% Yes

## Project Structure

```
Task-4-Bank-Marketing-Prediction/
│
├── bank_marketing_prediction.ipynb   # Main notebook
└── README.md                         # This file
```

## What This Notebook Covers

### 1. Exploratory Data Analysis
- Distribution plots for all numerical features split by target class
- Subscription rate by job type and education level
- Correlation heatmap across numerical features

### 2. Preprocessing & Feature Encoding
- Label encoding for target variable (`no → 0`, `yes → 1`)
- One-hot encoding for all categorical features
- Class imbalance handled using **SMOTE** (Synthetic Minority Oversampling)
- Feature scaling with `StandardScaler` for Logistic Regression

### 3. Model Training
Three classification models trained and evaluated:

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Bagged decision trees |
| Gradient Boosting | Boosted decision trees |

5-fold stratified cross-validation used for all models.

### 4. Evaluation
- Confusion Matrix
- F1-Score (primary metric due to class imbalance)
- ROC-AUC Curve
- Full classification report (precision, recall, support)

### 5. Feature Importance
Top 20 most important features ranked by Random Forest feature importances.

### 6. Explainable AI — SHAP
- **Beeswarm plot** — global view of how each feature impacts predictions
- **Waterfall plots** — 5 individual prediction explanations (True Positive, True Negative, False Positive, False Negative)
- **Dependence plot** — how the top feature affects model output across all samples

## Results Summary

| Model | F1-Score | ROC-AUC |
|---|---|---|
| Gradient Boosting | best | ~0.93 |
| Random Forest | 2nd | ~0.91 |
| Logistic Regression | baseline | ~0.86 |

> Exact scores will appear after running the notebook.

## How to Run

**Step 1 — Clone or open the repo in VS Code**

**Step 2 — Install dependencies** (run Cell 0 in the notebook, or paste this in terminal):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap imbalanced-learn ucimlrepo
```

**Step 3 — Run all cells top to bottom**

The dataset downloads automatically from UCI — no manual download needed.

## Skills Demonstrated

- Binary classification on imbalanced real-world data
- Categorical feature encoding strategies
- Handling class imbalance with SMOTE
- Model comparison and selection
- Explainable AI with SHAP (both global and local explanations)
- Customer behaviour analysis through feature importance


