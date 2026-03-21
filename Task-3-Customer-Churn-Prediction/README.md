# Customer Churn Prediction (Bank Customers)
### Binary Classification · Random Forest · Feature Importance · Label & One-Hot Encoding


## Overview

A complete **customer churn prediction project** that identifies which bank customers are likely to leave. The project covers the full machine learning pipeline — data cleaning, categorical encoding, exploratory data analysis, model training using Random Forest, and feature importance analysis to understand the real drivers of churn.

Built entirely in a **Jupyter Notebook inside VS Code**, with every section documented using markdown for clarity and reproducibility.

## The Dataset

| Property       | Detail                                          |
|----------------|-------------------------------------------------|
| Source         | Kaggle — Churn Modelling Dataset                |
| Total Samples  | 10,000 bank customers                           |
| Features       | 14 columns (11 usable after dropping identifiers) |
| Target Column  | `Exited` — 1 (Churned) / 0 (Stayed)            |
| Missing Values | None                                            |
| Class Balance  | ~20% churned, ~80% stayed                      |

### Features

| Feature           | Type        | Description                                      |
|-------------------|-------------|--------------------------------------------------|
| CreditScore       | Numerical   | Customer credit score                            |
| Geography         | Categorical | Country — France, Germany, Spain                 |
| Gender            | Categorical | Male / Female                                    |
| Age               | Numerical   | Customer age in years                            |
| Tenure            | Numerical   | Number of years with the bank                    |
| Balance           | Numerical   | Account balance                                  |
| NumOfProducts     | Numerical   | Number of bank products held                     |
| HasCrCard         | Binary      | 1 = has credit card, 0 = does not               |
| IsActiveMember    | Binary      | 1 = active member, 0 = inactive                 |
| EstimatedSalary   | Numerical   | Estimated annual salary                          |
| Exited            | Binary      | **Target** — 1 = churned, 0 = stayed            |

> `RowNumber`, `CustomerId`, and `Surname` are dropped — they are identifiers with no predictive value.

## Stack

| Tool            | Role                                                  |
|-----------------|-------------------------------------------------------|
| `pandas`        | Data loading, cleaning, grouping, and encoding        |
| `numpy`         | Numerical computation                                 |
| `matplotlib`    | Base chart rendering and layouts                      |
| `seaborn`       | Statistical plots — heatmap, box plots, bar charts    |
| `scikit-learn`  | Encoding, scaling, model training, and evaluation     |

## Project Structure

```
customer-churn-prediction/
│
├── customer_churn_prediction.ipynb   ← Full notebook with markdown + code
├── Churn_Modelling.csv               ← Dataset from Kaggle (place here)
└── README.md                         ← This file
```

## Setup

**Clone and install:**

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Download the dataset:**

👉 [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

Place `Churn_Modelling.csv` in the same folder as the notebook before running.

**Run the notebook:**

Open `customer_churn_prediction.ipynb` in VS Code, select your Python interpreter, and run cells with `Shift + Enter`.

> Requires the **Jupyter extension** in VS Code.

---

## Analysis Breakdown

### Data Cleaning
The dataset arrives in a clean state with no missing values. Three identifier columns — `RowNumber`, `CustomerId`, and `Surname` — are dropped before any analysis because they carry no behavioral information about the customer. Duplicate rows are checked and removed if present to prevent data leakage during model training.

### Exploratory Data Analysis

**Churn Distribution** — Approximately 20% of customers churned, creating a moderately imbalanced dataset. A naive classifier predicting "stayed" for every customer would achieve 80% accuracy — making accuracy alone a misleading metric. F1 Score and AUC are used alongside accuracy to properly assess model performance.

**Geography and Gender** — Customers from Germany churn at nearly double the rate of France and Spain, making geography one of the most visually striking risk factors. Female customers show a slightly higher churn tendency than male customers across all three countries.

**Age and Balance** — Churned customers skew significantly older — the age distribution of churned customers peaks in the 40–55 range while retained customers peak in the 30–40 range. High-balance customers also show elevated churn rates, representing the most financially impactful segment to lose.

**Engagement Features** — Inactive members churn at a substantially higher rate than active members, confirming that engagement is a strong protective factor. Credit card ownership shows minimal impact on churn rate.

**Correlation Heatmap** — Among numerical features, `Age` has the strongest positive correlation with `Exited`. `IsActiveMember` has the strongest negative correlation — active members are significantly less likely to leave.

### Categorical Encoding

Two different encoding strategies are applied depending on the nature of the column:

**Label Encoding — Gender**
Gender has exactly two values (Male / Female), making Label Encoding the correct and efficient choice. One column becomes `0` for Female and `1` for Male. One-Hot Encoding would be redundant here — knowing one value always reveals the other.

**One-Hot Encoding — Geography**
Geography has three values (France, Germany, Spain) with no natural numeric ordering between countries. One-Hot Encoding creates three binary columns — one per country. The France column is dropped using `drop_first=True` to avoid the dummy variable trap, where two columns together always perfectly predict the third.

### Model — Random Forest Classifier
Random Forest is an ensemble model that combines the predictions of 100 individual decision trees trained on random subsets of the data. Each tree votes, and the majority vote determines the final prediction. Key training parameters include `max_depth=10` to prevent overfitting on any single tree and `class_weight='balanced'` to compensate for the 80/20 class imbalance by giving churned customers proportionally more weight during training.

### Model Evaluation

Performance is measured across five metrics:

| Metric    | Description                                                      |
|-----------|------------------------------------------------------------------|
| Accuracy  | Overall percentage of correct predictions                        |
| Precision | Of all customers flagged as churners, how many actually churned? |
| Recall    | Of all actual churners, how many were successfully identified?   |
| F1 Score  | Harmonic mean of precision and recall                            |
| AUC       | Ability to separate churners from non-churners at all thresholds |

The **ROC Curve** plots the True Positive Rate against the False Positive Rate across all decision thresholds. An AUC above 0.80 confirms the model is meaningfully distinguishing churners from retained customers — not just predicting the majority class.

In churn prediction, **False Negatives carry the highest business cost** — a missed churner is a lost customer with no opportunity for retention intervention.

### Feature Importance
Random Forest generates an importance score for every feature based on how much each one reduced prediction error across all 100 trees. Features used frequently in early splits across many trees receive the highest scores.

Expected top features in this dataset:

| Feature         | Expected Importance | Business Meaning                             |
|-----------------|---------------------|----------------------------------------------|
| Age             | Highest             | Older customers are significantly more likely to churn |
| Balance         | High                | High-balance customers represent elevated churn risk   |
| NumOfProducts   | High                | Customers with 1 product have less engagement         |
| IsActiveMember  | Medium              | Inactive members churn more frequently               |
| Geography       | Medium              | Germany has a notably higher churn rate               |

---

## Key Findings

**1. Age is the strongest churn predictor.**
Older customers churn at a disproportionately higher rate. The age distribution of churned customers peaks roughly 10–15 years later than retained customers, suggesting that older demographics have different banking expectations or are more likely to consolidate accounts.

**2. Germany is a high-risk geography.**
Customers based in Germany churn at nearly twice the rate of those in France and Spain. This geographic disparity suggests country-specific factors — service quality, local competition, or product fit — are influencing retention.

**3. Inactive members are the most at-risk segment.**
`IsActiveMember = 0` correlates strongly with churn. A customer who is not engaging with the bank's products or services has little reason to stay when a competitor offers a better deal.

**4. High-balance customers churning is a priority concern.**
The balance distributions show that churned customers tend to carry higher account balances than retained ones. Losing high-balance customers has a disproportionate impact on the bank's assets under management.

**5. Number of products is a loyalty signal.**
Customers holding only one product churn significantly more than those with two or more. Cross-selling additional products appears to increase switching costs and strengthen customer retention.

## Statistical Summary

| Feature          | Mean      | Std Dev   | Min   | Max      |
|------------------|-----------|-----------|-------|----------|
| CreditScore      | 650.53    | 96.65     | 350   | 850      |
| Age              | 38.92     | 10.49     | 18    | 92       |
| Tenure           | 5.01      | 2.89      | 0     | 10       |
| Balance          | 76,485.89 | 62,397.41 | 0     | 250,898  |
| NumOfProducts    | 1.53      | 0.58      | 1     | 4        |
| EstimatedSalary  | 100,090.24| 57,510.49 | 11.58 | 199,992  |

---

## Author

**Momina Ramzan**

## Acknowledgements

- [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) — dataset source
- Scikit-learn contributors — model, encoding, and evaluation tools
