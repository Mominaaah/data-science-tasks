# Data Science Projects Portfolio

## About

This repository contains a series of data science projects built using Python and Jupyter Notebooks in VS Code.
Each project covers a different area of the data science workflow from exploratory analysis and visualization through to machine learning, unsupervised learning, time series forecasting, and explainable AI.
All projects are documented with step-by-step markdown explanations alongside every code cell.

---

## Projects

### Task 1 — Iris Dataset Explorer
**Exploratory Data Analysis · Statistical Visualization · Descriptive Statistics**

An end-to-end exploratory data analysis of the classic Iris dataset introduced by R. A. Fisher in 1936.
The project applies the full EDA pipeline — loading data, computing statistics, and generating multi-layered visualizations to extract meaningful insights about three iris flower species.

| Property  | Detail |
|---|---|
| Dataset   | Iris Dataset — UCI Machine Learning Repository |
| Samples   | 150 flowers across 3 species |
| Features  | 4 — Sepal Length, Sepal Width, Petal Length, Petal Width |
| Objective | Understand feature distributions and separability |

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

| Property  | Detail |
|---|---|
| Dataset   | Loan Prediction Dataset — Kaggle |
| Samples   | 614 loan applicants |
| Features  | 11 — income, loan amount, credit history, education, and more |
| Objective | Predict loan approval / rejection |

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

| Property  | Detail |
|---|---|
| Dataset   | Churn Modelling Dataset — Kaggle |
| Samples   | 10,000 bank customers |
| Features  | 11 — age, balance, geography, activity status, and more |
| Objective | Identify customers at risk of leaving the bank |

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

### Task 4 — Term Deposit Subscription Prediction
**Classification · Feature Encoding · SMOTE · Explainable AI (SHAP)**

A supervised learning project that predicts whether a bank customer will subscribe to a term deposit following a marketing campaign.
The project handles severe class imbalance using SMOTE, compares three classifiers, and uses SHAP to explain individual model predictions.

| Property  | Detail |
|---|---|
| Dataset   | Bank Marketing Dataset — UCI Machine Learning Repository |
| Samples   | 45,211 customer records |
| Features  | 16 — demographic, financial, and campaign contact data |
| Objective | Predict term deposit subscription (yes / no) |

**Techniques Applied:**
- One-Hot Encoding for all categorical features
- SMOTE — synthetic oversampling to fix class imbalance (11.7% → 50%)
- Logistic Regression, Random Forest, and Gradient Boosting classifiers
- 5-fold stratified cross-validation
- Confusion Matrix, F1-Score, and ROC-AUC evaluation
- SHAP TreeExplainer — global beeswarm + 5 individual waterfall explanations

**Key Findings:**
- `duration` (call duration) is the strongest predictor of subscription
- Gradient Boosting consistently outperforms the other two models on F1
- Customers contacted in March, September, and October subscribe at higher rates
- SHAP reveals that longer calls and previous successful contacts drive positive predictions

[View Project Folder](./Task-4-Bank-Marketing-Prediction) &nbsp;|&nbsp; [Open Notebook](./Task-4-Bank-Marketing-Prediction/bank_marketing_prediction.ipynb)

---

### Task 5 — Customer Segmentation Using Unsupervised Learning
**K-Means Clustering · PCA · t-SNE · Marketing Strategy Development**

An unsupervised learning project that clusters mall customers into distinct segments based on income and spending behaviour.
Each identified segment is profiled and matched with a data-driven marketing strategy.

| Property  | Detail |
|---|---|
| Dataset   | Mall Customers Dataset |
| Samples   | 200 customer records |
| Features  | 3 — Age, Annual Income (k$), Spending Score (1–100) |
| Objective | Segment customers and propose targeted marketing strategies |

**Techniques Applied:**
- Elbow Method, Silhouette Score, and Davies-Bouldin Score to find optimal K
- K-Means clustering (K = 5) on 2-D and 3-D feature spaces
- PCA — linear dimensionality reduction with scree plot and biplot
- t-SNE — non-linear dimensionality reduction for visual cluster validation
- Cluster profiling via mean feature values and violin plots
- Marketing strategy design per segment based on income × spending behaviour

**Segments Identified:**

| Cluster | Segment | Strategy |
|---|---|---|
| 0 | High Income, Low Spending | Premium loyalty + trust building |
| 1 | Medium Income, Medium Spending | Rewards, bundles, social proof |
| 2 | High Income, High Spending | VIP retention + luxury positioning |
| 3 | Low Income, High Spending | BNPL, influencer marketing, flash sales |
| 4 | Low Income, Low Spending | Essential products, heavy discounts |

[View Project Folder](./Task-5-Customer-Segmentation) &nbsp;|&nbsp; [Open Notebook](./Task-5-Customer-Segmentation/customer_segmentation.ipynb)

---

### Task 6 — Energy Consumption Time Series Forecasting
**Time Series · Feature Engineering · ARIMA · Prophet · XGBoost · MAE · RMSE**

A time series forecasting project that predicts short-term household energy usage from over 2 million minute-level power readings.
Three models are trained and compared — a classical statistical model, a decomposition-based model, and a machine learning model.

| Property  | Detail |
|---|---|
| Dataset   | Individual Household Electric Power Consumption — UCI |
| Samples   | ~2M rows resampled to ~35,000 hourly records |
| Features  | Global Active Power (kW) + engineered time/lag features |
| Objective | Forecast the next 7 days of energy usage |

**Techniques Applied:**
- Resampling from 1-minute to hourly resolution
- Augmented Dickey-Fuller stationarity test
- ACF and PACF analysis for lag structure identification
- Feature engineering — calendar flags, lag features (1h, 24h, 168h), rolling averages
- ARIMA(2,1,2) — classical statistical forecasting
- Prophet — trend + daily/weekly seasonality decomposition
- XGBoost — gradient boosted trees on engineered features
- MAE and RMSE evaluation on a 7-day held-out test set

**Key Findings:**
- XGBoost achieves the lowest MAE by leveraging lag and rolling features
- Prophet captures weekly seasonality patterns cleanly but struggles with sudden spikes
- `lag_24` and `lag_168` are the most important XGBoost features — same hour yesterday and same hour last week
- Energy usage peaks in the early evening (6–9 PM) and drops sharply after midnight

> **Note:** The dataset file (124 MB) is not included in this repo. Download it from the UCI link below.

[View Project Folder](./Task-6-Energy_Forecasting) &nbsp;|&nbsp; [Open Notebook](./Task-6-Energy_Forecasting/energy_forecasting.ipynb)

---

## Skills Demonstrated

| Area | Tools and Techniques |
|---|---|
| Data Loading | `pandas.read_csv`, `sklearn.datasets`, `ucimlrepo` |
| Data Cleaning | Null detection, median/mode imputation, duplicate removal |
| Exploratory Analysis | Histograms, scatter plots, box plots, pair plots, heatmaps |
| Statistical Testing | Pearson correlation, one-way ANOVA, ADF stationarity test |
| Categorical Encoding | `LabelEncoder`, `pd.get_dummies` (One-Hot Encoding) |
| Feature Scaling | `StandardScaler` |
| Class Imbalance | SMOTE oversampling |
| Feature Engineering | Lag features, rolling statistics, calendar flags |
| Machine Learning | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| Unsupervised Learning | K-Means Clustering |
| Dimensionality Reduction | PCA, t-SNE |
| Time Series Forecasting | ARIMA, Prophet |
| Model Evaluation | Accuracy, Precision, Recall, F1, AUC, ROC Curve, MAE, RMSE |
| Explainable AI | SHAP (TreeExplainer, beeswarm, waterfall, dependence plots) |
| Visualization | `matplotlib`, `seaborn`, `shap`, `ConfusionMatrixDisplay` |

---

## Repository Structure

```
data-science-tasks/
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
├── Task-4-Bank-Marketing-Prediction/
│   ├── bank_marketing_prediction.ipynb
│   └── README.md
│
├── Task-5-Customer-Segmentation/
│   ├── customer_segmentation.ipynb
│   └── README.md
│
├── Task-6-Energy_Forecasting/
│   ├── energy_forecasting.ipynb
│   └── README.md
│
└── README.md             ← This file
```

---

## Setup

**Clone the repository:**

```bash
git clone https://github.com/Mominaaah/data-science-tasks.git
cd data-science-tasks
```

**Install all dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels prophet xgboost shap imbalanced-learn ucimlrepo
```

**Open any notebook in VS Code:**

1. Open the project folder in VS Code
2. Open the `.ipynb` file for the task you want to run
3. Select your Python interpreter (top right of the notebook)
4. Run cells with `Shift + Enter` or use `Kernel → Restart & Run All`

> Requires the **Jupyter extension** in VS Code.

---

## Datasets

| Task   | Dataset | Source | Link |
|---|---|---|---|
| Task 1 | Iris Dataset | Built-in (`sklearn.datasets`) | No download needed |
| Task 2 | Loan Prediction Dataset | Kaggle | [Download](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) |
| Task 3 | Churn Modelling Dataset | Kaggle | [Download](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| Task 4 | Bank Marketing Dataset | UCI | [Download](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Task 5 | Mall Customers Dataset | Kaggle | [Download](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| Task 6 | Household Power Consumption | UCI | [Download](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) |

---

## Author

**Momina Ramzan**  
GitHub · [@Mominaaah](https://github.com/Mominaaah)
