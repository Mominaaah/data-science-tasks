# Data Science Projects Portfolio

## About

This repository contains a series of data science projects built using Python and Jupyter Notebooks in VS Code.
Each project covers a different area of the data science workflow — from exploratory analysis and visualization through to machine learning model training and evaluation.
All projects are documented with step-by-step markdown explanations alongside every code cell.

## Projects

### Task 1 — Iris Dataset Explorer
**Exploratory Data Analysis · Statistical Visualization · Descriptive Statistics**

An end-to-end exploratory data analysis of the classic Iris dataset introduced by R. A. Fisher in 1936.
The project applies the full EDA pipeline — loading data, computing statistics, and generating multi-layered visualizations to extract meaningful insights about three iris flower species.

| Property       | Detail                                      |
|----------------|---------------------------------------------|
| Dataset        | Iris Dataset — UCI Machine Learning Repository |
| Samples        | 150 flowers across 3 species                |
| Features       | 4 — Sepal Length, Sepal Width, Petal Length, Petal Width |
| Objective      | Understand feature distributions and separability |

**Techniques Applied:**
- Descriptive statistics — mean, median, standard deviation per species
- Scatter plots, pair plots, histograms, box plots
- Pearson correlation heatmap
- One-way ANOVA — statistical significance testing across species

**Key Findings:**
- Petal features separate species far more cleanly than sepal features
- Iris Setosa is perfectly linearly separable from the other two species
- Petal Length and Petal Width share a near-perfect correlation of 0.96
- All four features pass the ANOVA test — every feature is statistically significant

[View Project Folder](./Task-1-Iris-Analysis) &nbsp;|&nbsp; [Open Notebook](./Task-1-Iris-Analysis/iris_notebook.ipynb)

---

### Task 2 — Credit Risk Prediction
**Binary Classification · Logistic Regression · Missing Value Handling · Model Evaluation**

A machine learning project that predicts whether a loan applicant is likely to default.
The project handles real-world data quality issues including missing values, applies feature encoding, trains a Logistic Regression classifier, and evaluates it using accuracy, confusion matrix, and classification metrics.

| Property       | Detail                                      |
|----------------|---------------------------------------------|
| Dataset        | Loan Prediction Dataset — Kaggle            |
| Samples        | 614 loan applicants                         |
| Features       | 11 — income, loan amount, credit history, education, and more |
| Objective      | Predict loan approval / rejection           |

**Techniques Applied:**
- Missing value imputation — median for numerical, mode for categorical columns
- Label Encoding for all categorical features
- Logistic Regression — binary classification baseline model
- Confusion matrix, precision, recall, F1 score evaluation

**Key Findings:**
- `Credit_History` is the single strongest predictor of loan approval
- Applicants with no credit history are rejected at a dramatically higher rate
- Income alone is not a reliable separator between approved and rejected applicants
- The dataset is moderately imbalanced — 69% approved vs 31% rejected

[View Project Folder](./Task-2-Credit-Risk-Prediction) &nbsp;|&nbsp; [Open Notebook](./Task-2-Credit-Risk-Prediction/credit_risk_prediction.ipynb)

---

### Task 3 — Customer Churn Prediction
**Random Forest · Feature Importance · One-Hot Encoding · ROC Curve · AUC**

A classification project that identifies which bank customers are likely to leave.
The project demonstrates both Label Encoding and One-Hot Encoding strategies, trains a Random Forest classifier with balanced class weights, and uses feature importance analysis to uncover the real drivers of customer churn.

| Property       | Detail                                      |
|----------------|---------------------------------------------|
| Dataset        | Churn Modelling Dataset — Kaggle            |
| Samples        | 10,000 bank customers                       |
| Features       | 11 — age, balance, geography, activity status, and more |
| Objective      | Identify customers at risk of leaving the bank |

**Techniques Applied:**
- Label Encoding for binary categorical column (Gender)
- One-Hot Encoding for multi-class categorical column (Geography)
- StandardScaler — feature normalization before model training
- Random Forest Classifier — 100 trees with balanced class weights
- ROC Curve and AUC Score alongside confusion matrix and F1 evaluation
- Feature importance analysis — ranking all features by their predictive contribution

**Key Findings:**
- `Age` is the strongest churn predictor — older customers churn significantly more
- German customers churn at nearly double the rate of France and Spain
- Inactive members are the highest-risk segment regardless of balance or salary
- Customers holding only one product churn far more than multi-product customers
- High-balance customers churning represents the greatest financial risk to the bank

[View Project Folder](./Task-3-Customer-Churn-Prediction) &nbsp;|&nbsp; [Open Notebook](./Task-3-Customer-Churn-Prediction/customer_churn_prediction.ipynb)

---

## Skills Demonstrated

| Area                        | Tools and Techniques                                      |
|-----------------------------|-----------------------------------------------------------|
| Data Loading                | `pandas.read_csv`, `sklearn.datasets`                     |
| Data Cleaning               | Null detection, median/mode imputation, duplicate removal |
| Exploratory Analysis        | Histograms, scatter plots, box plots, pair plots          |
| Statistical Testing         | Pearson correlation, one-way ANOVA                        |
| Categorical Encoding        | `LabelEncoder`, `pd.get_dummies` (One-Hot Encoding)       |
| Feature Scaling             | `StandardScaler`                                          |
| Machine Learning Models     | Logistic Regression, Random Forest Classifier             |
| Model Evaluation            | Accuracy, Precision, Recall, F1 Score, AUC, ROC Curve    |
| Feature Importance          | Random Forest `feature_importances_`                      |
| Visualization               | `matplotlib`, `seaborn`, `ConfusionMatrixDisplay`         |

---

## Repository Structure

```
 
 data-science-portfolio/
│
├── Task-1-Iris-Analysis/
│   ├── iris_notebook.ipynb
│   └── README.md
│
├── Task-2-Credit-Risk-Prediction/
│   ├── credit_risk_prediction.ipynb
│   ├── train.csv
│   └── README.md
│
├── Task-3-Customer-Churn-Prediction/
│   ├── customer_churn_prediction.ipynb
│   ├── Churn_Modelling.csv
│   └── README.md
│
└── README.md             ← This file
```

---

## Setup

**Clone the repository:**

```bash
git clone https://github.com/yourusername/data-science-tasks.git
cd data-science-tasks
```

**Install all dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

**Open any notebook in VS Code:**

1. Open the project folder in VS Code
2. Open the `.ipynb` file for the task you want to run
3. Select your Python interpreter (top right of the notebook)
4. Run cells with `Shift + Enter` or use `Kernel → Restart & Run All`

> Requires the **Jupyter extension** in VS Code.

## Datasets

| Task   | Dataset                   | Source  | Link |
|--------|---------------------------|---------|------|
| Task 1 | Iris Dataset              | Built-in (`sklearn.datasets`) | No download needed |
| Task 2 | Loan Prediction Dataset   | Kaggle  | [Download](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) |
| Task 3 | Churn Modelling Dataset   | Kaggle  | [Download](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |

---

## Author

**Momina Ramzan**
GitHub · [@Mominaaah](https://github.com/Mominaaah)

